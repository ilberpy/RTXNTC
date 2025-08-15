/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "ForwardShadingCommon.hlsli"

#define MATERIAL_REGISTER_SPACE     FORWARD_SPACE_MATERIAL
#define MATERIAL_CB_SLOT            FORWARD_BINDING_MATERIAL_CONSTANTS
#define MATERIAL_DIFFUSE_SLOT       FORWARD_BINDING_MATERIAL_DIFFUSE_TEXTURE
#define MATERIAL_SPECULAR_SLOT      FORWARD_BINDING_MATERIAL_SPECULAR_TEXTURE
#define MATERIAL_NORMALS_SLOT       FORWARD_BINDING_MATERIAL_NORMAL_TEXTURE
#define MATERIAL_EMISSIVE_SLOT      FORWARD_BINDING_MATERIAL_EMISSIVE_TEXTURE
#define MATERIAL_OCCLUSION_SLOT     FORWARD_BINDING_MATERIAL_OCCLUSION_TEXTURE
#define MATERIAL_TRANSMISSION_SLOT  FORWARD_BINDING_MATERIAL_TRANSMISSION_TEXTURE
#define MATERIAL_OPACITY_SLOT       FORWARD_BINDING_MATERIAL_OPACITY_TEXTURE

#define MATERIAL_SAMPLER_REGISTER_SPACE FORWARD_SPACE_SHADING
#define MATERIAL_SAMPLER_SLOT           FORWARD_BINDING_MATERIAL_SAMPLER

#include "donut/shaders/material_bindings.hlsli"
#include "donut/shaders/hash_based_rng.hlsli"

#include "NtcForwardShadingPassConstants.h"

DECLARE_CBUFFER(ForwardShadingViewConstants, g_ForwardView, FORWARD_BINDING_VIEW_CONSTANTS, FORWARD_SPACE_VIEW);
DECLARE_CBUFFER(ForwardShadingLightConstants, g_ForwardLight, FORWARD_BINDING_LIGHT_CONSTANTS, FORWARD_SPACE_SHADING);
DECLARE_CBUFFER(NtcForwardShadingPassConstants, g_Pass, FORWARD_BINDING_NTC_PASS_CONSTANTS, FORWARD_SPACE_SHADING);
SamplerState s_StfSampler : REGISTER_SAMPLER(FORWARD_BINDING_STF_SAMPLER,   FORWARD_SPACE_SHADING);

FeedbackTexture2D<SAMPLER_FEEDBACK_MIN_MIP> t_BaseOrDiffuseFeedback         : REGISTER_UAV(FORWARD_BINDING_MATERIAL_DIFFUSE_FEEDBACK_UAV, FORWARD_SPACE_MATERIAL);
FeedbackTexture2D<SAMPLER_FEEDBACK_MIN_MIP> t_MetalRoughOrSpecularFeedback  : REGISTER_UAV(FORWARD_BINDING_MATERIAL_SPECULAR_FEEDBACK_UAV, FORWARD_SPACE_MATERIAL);
FeedbackTexture2D<SAMPLER_FEEDBACK_MIN_MIP> t_NormalFeedback                : REGISTER_UAV(FORWARD_BINDING_MATERIAL_NORMAL_FEEDBACK_UAV, FORWARD_SPACE_MATERIAL);
FeedbackTexture2D<SAMPLER_FEEDBACK_MIN_MIP> t_EmissiveFeedback              : REGISTER_UAV(FORWARD_BINDING_MATERIAL_EMISSIVE_FEEDBACK_UAV, FORWARD_SPACE_MATERIAL);
FeedbackTexture2D<SAMPLER_FEEDBACK_MIN_MIP> t_OcclusionFeedback             : REGISTER_UAV(FORWARD_BINDING_MATERIAL_OCCLUSION_FEEDBACK_UAV, FORWARD_SPACE_MATERIAL);
FeedbackTexture2D<SAMPLER_FEEDBACK_MIN_MIP> t_TransmissionFeedback          : REGISTER_UAV(FORWARD_BINDING_MATERIAL_TRANSMISSION_FEEDBACK_UAV, FORWARD_SPACE_MATERIAL);
FeedbackTexture2D<SAMPLER_FEEDBACK_MIN_MIP> t_OpacityFeedback               : REGISTER_UAV(FORWARD_BINDING_MATERIAL_OPACITY_FEEDBACK_UAV, FORWARD_SPACE_MATERIAL);

#if USE_STF

#define STF_SHADER_STAGE STF_SHADER_STAGE_PIXEL
#define STF_SHADER_MODEL_MAJOR 6
#define STF_SHADER_MODEL_MINOR 6
#include "STFSamplerState.hlsli"

float4 SampleTextureWithSTF(Texture2D texture, float4 random, float2 uv, uint mipBias, out uint status)
{
    STF_SamplerState sampler = STF_SamplerState::Create(random);
    sampler.SetAnisoMethod(STF_ANISO_LOD_METHOD_DEFAULT);
    sampler.SetFilterType(g_Pass.stfFilterMode);

    int2 textureSize;
    int mipLevels;
    texture.GetDimensions(mipBias, textureSize.x, textureSize.y, mipLevels);
    float3 samplePos = sampler.Texture2DGetSamplePos(textureSize.x, textureSize.y, mipLevels, uv);
    int mipLevel = int(samplePos.z) + int(mipBias);
    
    int2 offsetZero = int2(0, 0);
    return texture.SampleLevel(s_StfSampler, samplePos.xy, mipLevel, offsetZero, status);
}

// A version of SampleWithFeedback that uses STF to sample the texture
float4 SampleWithFeedbackSTF(float2 texCoord, Texture2D tex, FeedbackTexture2D<SAMPLER_FEEDBACK_MIN_MIP> texFeedback,
    bool enableFeedback, float4 random)
{
    uint status;
    float4 col = SampleTextureWithSTF(tex, random, texCoord, 0, status); // Opportunistic sample
    if (!CheckAccessFullyMapped(status))
    {
        uint level = tex.CalculateLevelOfDetail(s_MaterialSampler, texCoord);
        for (uint i = level + 1; i < 16; ++i)
        {
            // When going to coarser MIPs, we need to recalculate the STF sample position because
            // those MIPs have smaller dimensions, and therefore the sample footprint is larger in UV space
            col = SampleTextureWithSTF(tex, random, texCoord, i, status);
            if (CheckAccessFullyMapped(status))
                break;
        }
    }

    if (enableFeedback)
        texFeedback.WriteSamplerFeedback(tex, s_MaterialSampler, texCoord);

    return col;
}

MaterialTextureSample SampleMaterialTexturesFeedbackSTF(uint2 pixelPosition, float2 texCoord, bool enableFeedback)
{
    HashBasedRNG rng = HashBasedRNG::Create2D(pixelPosition, g_Pass.frameIndex);
    float4 random = rng.NextFloat4();

    MaterialTextureSample values = DefaultMaterialTextures();

    if (g_Material.flags & MaterialFlags_UseBaseOrDiffuseTexture)
    {
        values.baseOrDiffuse = SampleWithFeedbackSTF(texCoord, t_BaseOrDiffuse, t_BaseOrDiffuseFeedback, enableFeedback, random);
    }

    if (g_Material.flags & MaterialFlags_UseMetalRoughOrSpecularTexture)
    {
        values.metalRoughOrSpecular = SampleWithFeedbackSTF(texCoord, t_MetalRoughOrSpecular, t_MetalRoughOrSpecularFeedback, enableFeedback, random);
    }

    if (g_Material.flags & MaterialFlags_UseEmissiveTexture)
    {
        values.emissive = SampleWithFeedbackSTF(texCoord, t_Emissive, t_EmissiveFeedback, enableFeedback, random);
    }

    if (g_Material.flags & MaterialFlags_UseNormalTexture)
    {
        values.normal = SampleWithFeedbackSTF(texCoord, t_Normal, t_NormalFeedback, enableFeedback, random);
    }

    if (g_Material.flags & MaterialFlags_UseOcclusionTexture)
    {
        values.occlusion = SampleWithFeedbackSTF(texCoord, t_Occlusion, t_OcclusionFeedback, enableFeedback, random);
    }

    if (g_Material.flags & MaterialFlags_UseTransmissionTexture)
    {
        values.transmission = SampleWithFeedbackSTF(texCoord, t_Transmission, t_TransmissionFeedback, enableFeedback, random);
    }

    if (g_Material.flags & MaterialFlags_UseOpacityTexture)
    {
        values.opacity = SampleWithFeedbackSTF(texCoord, t_Opacity, t_OpacityFeedback, enableFeedback, random).r;
    }

    return values;
}
#endif

float4 SampleWithFeedback(float2 texCoord, Texture2D tex, FeedbackTexture2D<SAMPLER_FEEDBACK_MIN_MIP> texFeedback, bool enableFeedback)
{
    int2 offsetZero = int2(0, 0);
    uint status;
    float4 col = tex.Sample(s_MaterialSampler, texCoord, offsetZero, 0, status); // Opportunistic sample
    if (!CheckAccessFullyMapped(status))
    {
        uint level = tex.CalculateLevelOfDetail(s_MaterialSampler, texCoord);
        for (uint i = level + 1; i < 16; ++i)
        {
            col = tex.Sample(s_MaterialSampler, texCoord, offsetZero, i, status);
            if (CheckAccessFullyMapped(status))
                break;
        }
    }

    if (enableFeedback)
        texFeedback.WriteSamplerFeedback(tex, s_MaterialSampler, texCoord);

    return col;
}

MaterialTextureSample SampleMaterialTexturesFeedback(float2 texCoord, bool enableFeedback)
{
    MaterialTextureSample values = DefaultMaterialTextures();

    if (g_Material.flags & MaterialFlags_UseBaseOrDiffuseTexture)
    {
        values.baseOrDiffuse = SampleWithFeedback(texCoord, t_BaseOrDiffuse, t_BaseOrDiffuseFeedback, enableFeedback);
    }

    if (g_Material.flags & MaterialFlags_UseMetalRoughOrSpecularTexture)
    {
        values.metalRoughOrSpecular = SampleWithFeedback(texCoord, t_MetalRoughOrSpecular, t_MetalRoughOrSpecularFeedback, enableFeedback);
    }

    if (g_Material.flags & MaterialFlags_UseEmissiveTexture)
    {
        values.emissive = SampleWithFeedback(texCoord, t_Emissive, t_EmissiveFeedback, enableFeedback);
    }

    if (g_Material.flags & MaterialFlags_UseNormalTexture)
    {
        values.normal = SampleWithFeedback(texCoord, t_Normal, t_NormalFeedback, enableFeedback);
    }

    if (g_Material.flags & MaterialFlags_UseOcclusionTexture)
    {
        values.occlusion = SampleWithFeedback(texCoord, t_Occlusion, t_OcclusionFeedback, enableFeedback);
    }

    if (g_Material.flags & MaterialFlags_UseTransmissionTexture)
    {
        values.transmission = SampleWithFeedback(texCoord, t_Transmission, t_TransmissionFeedback, enableFeedback);
    }

    if (g_Material.flags & MaterialFlags_UseOpacityTexture)
    {
        values.opacity = SampleWithFeedback(texCoord, t_Opacity, t_OpacityFeedback, enableFeedback).r;
    }

    return values;
}

#if !ENABLE_ALPHA_TEST
[earlydepthstencil]
#endif
void main(
    in float4 i_position : SV_Position,
    in SceneVertex i_vtx,
    in bool i_isFrontFace : SV_IsFrontFace,
    out float4 o_color : SV_Target0
#if TRANSMISSIVE_MATERIAL
    , out float4 o_backgroundBlendFactor : SV_Target1
#endif
)
{
    bool enableFeedback = true;
    if (g_Pass.feedbackThreshold < 1.0)
    {
        uint2 uniformQuadCoord = uint2(i_position.xy) / 2;

        HashBasedRNG rng = HashBasedRNG::Create2D(uniformQuadCoord, g_Pass.frameIndex);
        enableFeedback = rng.NextFloat() < g_Pass.feedbackThreshold;
    }

#if USE_STF
    MaterialTextureSample textures = SampleMaterialTexturesFeedbackSTF(int2(i_position.xy), i_vtx.texCoord, enableFeedback);
#else
    MaterialTextureSample textures = SampleMaterialTexturesFeedback(i_vtx.texCoord, enableFeedback);
#endif

    MaterialConstants materialConstants = g_Material;
    MaterialSample surfaceMaterial = EvaluateSceneMaterial(i_vtx.normal, i_vtx.tangent, materialConstants, textures);
    float3 surfaceWorldPos = i_vtx.pos;

    EvaluateForwardShading(
        materialConstants,
        surfaceMaterial,
        i_vtx.pos,
        i_isFrontFace,
        g_ForwardView.view,
        g_ForwardLight,
        o_color
#if TRANSMISSIVE_MATERIAL
        , o_backgroundBlendFactor
#endif
    );
}
