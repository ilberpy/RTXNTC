/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#pragma pack_matrix(row_major)

#include "ModelViewConstants.h"
#include <libntc/shaders/ColorSpaces.hlsli>
#include <donut/shaders/binding_helpers.hlsli>
#include <donut/shaders/scene_material.hlsli>
#include <donut/shaders/lighting.hlsli>

ConstantBuffer<ModelViewConstants> g_Const : register(b0);

#ifdef __cplusplus
static const ModelViewConstants g_Const; // for Intellisense only
#endif

SamplerState s_InputSampler : register(s0);

VK_BINDING(0, 1) Texture2D t_BindlessTextures[] : register(t0, space1);

void MainVS(
    in uint i_vertex : SV_VertexID,
    out float4 o_clipPos : SV_Position,
    out float3 o_worldPos : WORLDPOS,
    out float3 o_tangent : TANGENT,
    out float2 o_uv : UV)
{
    uint u = i_vertex & 1;
    uint v = (i_vertex >> 1) & 1;

    float4 worldPos = float4(float(u) * 2 - 1, 0, float(v) * 2 - 1, 1);
    o_clipPos = mul(worldPos, g_Const.view.matWorldToClip);
    o_worldPos = worldPos.xyz;
    o_tangent = float3(1, 0, 0);
    o_uv = float2(u, v);
}

float GetFloat(float4 channels, int firstChannel)
{
    switch(firstChannel)
    {
        case 0: return channels.r;
        case 1: return channels.g;
        case 2: return channels.b;
        case 3: return channels.a;
        default: return 0;
    }
}

float3 GetFloat3(float4 channels, int firstChannel)
{
    return firstChannel == 0 ? channels.rgb : channels.gba;
}

template<typename T>
void MaybeDecodeSRGB(inout T value, int textureIndex)
{
    // Convert a value from sRGB post-sampling.
    // This is wrong because the conversion should be done pre-sampling, but hardware only supports that
    // for RGBA8_UNORM textures, and not any other format. Notably, sometimes RGBA16_UNORM textures are also sRGB-encoded.
    if (g_Const.convertFromSrgbMask & (1U << textureIndex))
        value = NtcSrgbColorSpace::Decode(value);
}

float4 MainPS(
    in float4 i_windowPos : SV_Position,
    in float3 i_worldPos : WORLDPOS,
    in float3 i_tangent : TANGENT,
    in float2 i_uv : UV) : SV_Target0
{
    const float3 incidentDirection = normalize(i_worldPos - g_Const.view.cameraDirectionOrPosition.xyz);

    MaterialSample materialSample = DefaultMaterialSample();
    materialSample.geometryNormal = normalize(cross(ddx(i_worldPos), ddy(i_worldPos)));
    if (dot(materialSample.geometryNormal, incidentDirection) > 0)
        materialSample.geometryNormal = -materialSample.geometryNormal;
    materialSample.shadingNormal = materialSample.geometryNormal;
    materialSample.roughness = 0.3; // TODO: make this adjustable
    materialSample.hasMetalRoughParams = true;

    int indexOffset = (g_Const.enableSplitScreen && int(i_windowPos.x) > g_Const.splitPosition) ? g_Const.decompressedTextureOffset : 0;

    if (g_Const.albedoTexture >= 0)
    {
        materialSample.baseColor = GetFloat3(t_BindlessTextures[g_Const.albedoTexture + indexOffset].SampleLevel(s_InputSampler, i_uv, g_Const.mipLevel), g_Const.albedoChannel);
        MaybeDecodeSRGB<float3>(materialSample.baseColor, g_Const.albedoTexture);
    }

    if (g_Const.alphaTexture >= 0)
    {
        materialSample.opacity = GetFloat(t_BindlessTextures[g_Const.alphaTexture + indexOffset].SampleLevel(s_InputSampler, i_uv, g_Const.mipLevel), g_Const.alphaChannel);
        materialSample.opacity = saturate(materialSample.opacity);
    }

    if (g_Const.emissiveTexture >= 0)
    {
        materialSample.emissiveColor = GetFloat3(t_BindlessTextures[g_Const.emissiveTexture + indexOffset].SampleLevel(s_InputSampler, i_uv, g_Const.mipLevel), g_Const.emissiveChannel);
        MaybeDecodeSRGB<float3>(materialSample.emissiveColor, g_Const.emissiveTexture);
    }

    if (g_Const.metalnessTexture >= 0)
    {
        materialSample.metalness = GetFloat(t_BindlessTextures[g_Const.metalnessTexture + indexOffset].SampleLevel(s_InputSampler, i_uv, g_Const.mipLevel), g_Const.metalnessChannel);
        MaybeDecodeSRGB<float>(materialSample.metalness, g_Const.metalnessTexture);
    }

    if (g_Const.normalTexture >= 0)
    {
        float3 normalMapValue = GetFloat3(t_BindlessTextures[g_Const.normalTexture + indexOffset].SampleLevel(s_InputSampler, i_uv, g_Const.mipLevel), g_Const.normalChannel);
        MaybeDecodeSRGB<float3>(normalMapValue, g_Const.normalTexture);
        ApplyNormalMap(materialSample, float4(i_tangent, 1.0), float4(normalMapValue, 0.0), 1.0);
    }

    if (g_Const.occlusionTexture >= 0)
    {
        materialSample.occlusion = GetFloat(t_BindlessTextures[g_Const.occlusionTexture + indexOffset].SampleLevel(s_InputSampler, i_uv, g_Const.mipLevel), g_Const.occlusionChannel);
        MaybeDecodeSRGB<float>(materialSample.occlusion, g_Const.occlusionTexture);
    }

    if (g_Const.roughnessTexture >= 0)
    {
        materialSample.roughness = GetFloat(t_BindlessTextures[g_Const.roughnessTexture + indexOffset].SampleLevel(s_InputSampler, i_uv, g_Const.mipLevel), g_Const.roughnessChannel);
        MaybeDecodeSRGB<float>(materialSample.roughness, g_Const.roughnessTexture);
    }
    
    ConvertMetalRoughToSpecularGloss(materialSample.baseColor, materialSample.metalness, materialSample.diffuseAlbedo, materialSample.specularF0);
    
    float3 diffuseRadiance, specularRadiance;
    ShadeSurface(g_Const.light, materialSample, i_worldPos, incidentDirection, diffuseRadiance, specularRadiance);

    const float3 ambientLighting = lerp(g_Const.groundColor, g_Const.skyColor, saturate(materialSample.shadingNormal.y * 0.5 + 0.5));
    
    const float3 finalColor = 
        diffuseRadiance + 
        specularRadiance + 
        materialSample.emissiveColor + 
        materialSample.diffuseAlbedo * materialSample.occlusion * ambientLighting;

    return float4(finalColor, materialSample.opacity);
}


VK_PUSH_CONSTANT ConstantBuffer<OverlayConstants> g_OverlayConst : register(b0);

float4 OverlayPS(
    in float4 i_windowPos : SV_Position,
    in float2 i_quadUv : UV) : SV_Target0
{
    const int distanceFromSplit = abs(int(i_windowPos.x) - g_OverlayConst.splitPosition);
    
    if (distanceFromSplit < 2)
    {
        float color = (distanceFromSplit == 0) ? 1.0 : 0.0; // Black bar with white in the middle
        return float4(color.xxx, 1.0);
    }
    else
    {
        discard;
        return 0;
    }
}