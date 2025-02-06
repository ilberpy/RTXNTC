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

#include "NtcForwardShadingPassConstants.h"

DECLARE_CBUFFER(ForwardShadingViewConstants, g_ForwardView, FORWARD_BINDING_VIEW_CONSTANTS, FORWARD_SPACE_VIEW);
DECLARE_CBUFFER(ForwardShadingLightConstants, g_ForwardLight, FORWARD_BINDING_LIGHT_CONSTANTS, FORWARD_SPACE_SHADING);
DECLARE_CBUFFER(NtcForwardShadingPassConstants, g_Pass, FORWARD_BINDING_NTC_PASS_CONSTANTS, FORWARD_SPACE_SHADING);
SamplerState s_StfSampler : REGISTER_SAMPLER(FORWARD_BINDING_STF_SAMPLER,   FORWARD_SPACE_SHADING);

#if USE_STF

#define STF_SHADER_STAGE STF_SHADER_STAGE_PIXEL
#define STF_SHADER_MODEL_MAJOR 6
#define STF_SHADER_MODEL_MINOR 6
#include "STFSamplerState.hlsli"

float4 SampleTextureWithSTF(Texture2D texture, float4 random, float2 uv)
{
    STF_SamplerState sampler = STF_SamplerState::Create(random);
    sampler.SetAnisoMethod(STF_ANISO_LOD_METHOD_DEFAULT);
    sampler.SetFilterType(g_Pass.stfFilterMode);

    int2 textureSize;
    int mipLevels;
    texture.GetDimensions(0, textureSize.x, textureSize.y, mipLevels);
    float3 samplePos = sampler.Texture2DGetSamplePos(textureSize.x, textureSize.y, mipLevels, uv);
    int mipLevel = int(samplePos.z);
    
    return texture.SampleLevel(s_StfSampler, samplePos.xy, mipLevel);
}

MaterialTextureSample SampleMaterialTexturesSTF(uint2 pixelPosition, float2 uv)
{
    HashBasedRNG rng = HashBasedRNG::Create2D(pixelPosition, g_Pass.frameIndex);
    float4 random = rng.NextFloat4();

    MaterialTextureSample values = DefaultMaterialTextures();

    if ((g_Material.flags & MaterialFlags_UseBaseOrDiffuseTexture) != 0)
    {
        values.baseOrDiffuse = SampleTextureWithSTF(t_BaseOrDiffuse, random, uv);
    }

    if ((g_Material.flags & MaterialFlags_UseMetalRoughOrSpecularTexture) != 0)
    {
        values.metalRoughOrSpecular = SampleTextureWithSTF(t_MetalRoughOrSpecular, random, uv);
    }

    if ((g_Material.flags & MaterialFlags_UseEmissiveTexture) != 0)
    {
        values.emissive = SampleTextureWithSTF(t_Emissive, random, uv);
    }

    if ((g_Material.flags & MaterialFlags_UseNormalTexture) != 0)
    {
        values.normal = SampleTextureWithSTF(t_Normal, random, uv);
    }

    if ((g_Material.flags & MaterialFlags_UseOcclusionTexture) != 0)
    {
        values.occlusion = SampleTextureWithSTF(t_Occlusion, random, uv);
    }

    if ((g_Material.flags & MaterialFlags_UseTransmissionTexture) != 0)
    {
        values.transmission = SampleTextureWithSTF(t_Transmission, random, uv);
    }

    if ((g_Material.flags & MaterialFlags_UseOpacityTexture) != 0)
    {
        values.opacity = SampleTextureWithSTF(t_Opacity, random, uv).r;
    }

    return values;
}

#endif // USE_STF

void main(
    in float4 i_position : SV_Position,
    in SceneVertex i_vtx,
    in bool i_isFrontFace : SV_IsFrontFace,
    VK_LOCATION_INDEX(0, 0) out float4 o_color : SV_Target0
#if TRANSMISSIVE_MATERIAL
    , VK_LOCATION_INDEX(0, 1) out float4 o_backgroundBlendFactor : SV_Target1
#endif
)
{
#if USE_STF
    MaterialTextureSample textures = SampleMaterialTexturesSTF(int2(i_position.xy), i_vtx.texCoord);
#else
    MaterialTextureSample textures = SampleMaterialTexturesAuto(i_vtx.texCoord, g_Material.normalTextureTransformScale);
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
