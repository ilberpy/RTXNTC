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

// Include the constants header unconditionally so that NTC_NETWORK_UNKNOWN is always defined
#include "libntc/shaders/InferenceConstants.h"
// Include other NTC headers only if there is actually a texture to decompress
#if NETWORK_VERSION != NTC_NETWORK_UNKNOWN
#include "libntc/shaders/Inference.hlsli"
typedef NtcNetworkParams<NETWORK_VERSION> NtcParams;
#endif

#define STF_SHADER_STAGE STF_SHADER_STAGE_PIXEL
#define STF_SHADER_MODEL_MAJOR 6
#define STF_SHADER_MODEL_MINOR 6
#include "STFSamplerState.hlsli"
#include "NtcForwardShadingPassConstants.h"

DECLARE_CBUFFER(MaterialConstants, g_Material, FORWARD_BINDING_MATERIAL_CONSTANTS, FORWARD_SPACE_MATERIAL);
DECLARE_CBUFFER(ForwardShadingViewConstants, g_ForwardView, FORWARD_BINDING_VIEW_CONSTANTS, FORWARD_SPACE_VIEW);
DECLARE_CBUFFER(ForwardShadingLightConstants, g_ForwardLight, FORWARD_BINDING_LIGHT_CONSTANTS, FORWARD_SPACE_SHADING);
DECLARE_CBUFFER(NtcForwardShadingPassConstants, g_Pass, FORWARD_BINDING_NTC_PASS_CONSTANTS, FORWARD_SPACE_SHADING);

#if NETWORK_VERSION != NTC_NETWORK_UNKNOWN

DECLARE_CBUFFER(NtcTextureSetConstants, g_NtcMaterial, FORWARD_BINDING_NTC_MATERIAL_CONSTANTS, FORWARD_SPACE_MATERIAL);
ByteAddressBuffer t_InputFile    : REGISTER_SRV(FORWARD_BINDING_NTC_LATENTS_BUFFER, FORWARD_SPACE_MATERIAL);
ByteAddressBuffer t_WeightBuffer : REGISTER_SRV(FORWARD_BINDING_NTC_WEIGHTS_BUFFER, FORWARD_SPACE_MATERIAL);

#define CHANNEL_BASE_COLOR      0
#define CHANNEL_OPACITY         3
#define CHANNEL_METAL_ROUGH     4
#define CHANNEL_SPECULAR_COLOR  4
#define CHANNEL_GLOSSINESS      7
#define CHANNEL_NORMAL          8
#define CHANNEL_OCCLUSION       11
#define CHANNEL_EMISSIVE        12
#define CHANNEL_TRANSMISSION    15

void GetSamplePositionWithSTF(inout HashBasedRNG rng, float2 uv, out int2 texel, out int mipLevel)
{
    float4 random = rng.NextFloat4();
    STF_SamplerState sampler = STF_SamplerState::Create(random);
    sampler.SetAnisoMethod(STF_ANISO_LOD_METHOD_DEFAULT);
    sampler.SetFilterType(g_Pass.stfFilterMode);

    const int2 textureSize = NtcGetTextureDimensions(g_NtcMaterial, 0);
    const int mipLevels = NtcGetTextureMipLevels(g_NtcMaterial);
    float3 samplePos = sampler.Texture2DGetSamplePos(textureSize.x, textureSize.y, mipLevels, uv);
    mipLevel = int(samplePos.z);

    const int2 mipSize = NtcGetTextureDimensions(g_NtcMaterial, mipLevel);

    bool border;
    samplePos.xy = STF_ApplyAddressingMode2D(samplePos.xy, mipSize, STF_ADDRESS_MODE_WRAP, border);

    texel = int2(floor(samplePos.xy * mipSize));
}

MaterialTextureSample SampleNtcMaterial(uint2 pixelPosition, float2 uv)
{
    HashBasedRNG rng = HashBasedRNG::Create2D(pixelPosition, g_Pass.frameIndex);

    // Find out which texel to decompress.
    int mipLevel;
    int2 texel;
    GetSamplePositionWithSTF(rng, uv, texel, mipLevel);

    // The NtcSampleTextureSet... functions can convert all channels to linear color based on metadata stored
    // in the constant buffer. But that can be relatively slow if not optimized away by the driver.
    // Since we know the color spaces for all channels in advance, linearize explicitly below.
    const bool linearizeColorsOnSample = false;

    // Decompress the texel and get all the channels.
    float channels[NtcParams::OUTPUT_CHANNELS];
#ifdef USE_COOPVEC
    #if USE_FP8
        NtcSampleTextureSet_CoopVec_FP8<NETWORK_VERSION>(g_NtcMaterial, t_InputFile, 0,
            t_WeightBuffer, 0, texel, mipLevel, linearizeColorsOnSample, channels);
    #else
        NtcSampleTextureSet_CoopVec_Int8<NETWORK_VERSION>(g_NtcMaterial, t_InputFile, 0,
            t_WeightBuffer, 0, texel, mipLevel, linearizeColorsOnSample, channels);
    #endif
#else
    NtcSampleTextureSet<NETWORK_VERSION>(g_NtcMaterial, t_InputFile, 0,
        t_WeightBuffer, 0, texel, mipLevel, linearizeColorsOnSample, channels);
#endif

    // Initialize the 'textures' object with default values to avoid corruption when not all textures are present.
    MaterialTextureSample textures = DefaultMaterialTextures();

    // Distribute the NTC channels into the MaterialTextureSample's fields using a fixed mapping.
    // The same fixed mapping is used by the scene preparation script, ../tools/convert_gltf_materials.py
    
    if (NtcTextureSetHasChannels(g_NtcMaterial, CHANNEL_BASE_COLOR, 3))
    {
        textures.baseOrDiffuse.rgb = float3(
            channels[CHANNEL_BASE_COLOR + 0],
            channels[CHANNEL_BASE_COLOR + 1],
            channels[CHANNEL_BASE_COLOR + 2]);


        if (!linearizeColorsOnSample)
            textures.baseOrDiffuse.rgb = NtcSrgbColorSpace::Decode(textures.baseOrDiffuse.rgb);
    }

    if (NtcTextureSetHasChannels(g_NtcMaterial, CHANNEL_OPACITY, 1))
    {
        textures.opacity.r = channels[CHANNEL_OPACITY];
    }

    if ((g_Material.flags & MaterialFlags_UseSpecularGlossModel) != 0)
    {
        if (NtcTextureSetHasChannels(g_NtcMaterial, CHANNEL_SPECULAR_COLOR, 3))
        {
            textures.metalRoughOrSpecular.rgb = float3(
                channels[CHANNEL_SPECULAR_COLOR + 0],
                channels[CHANNEL_SPECULAR_COLOR + 1],
                channels[CHANNEL_SPECULAR_COLOR + 2]);
                
            if (!linearizeColorsOnSample)
                textures.metalRoughOrSpecular.rgb = NtcSrgbColorSpace::Decode(textures.metalRoughOrSpecular.rgb);
        }
        
        if (NtcTextureSetHasChannels(g_NtcMaterial, CHANNEL_GLOSSINESS, 1))
        {
            textures.metalRoughOrSpecular.a = channels[CHANNEL_GLOSSINESS];
        }
    }
    else
    {
        if (NtcTextureSetHasChannels(g_NtcMaterial, CHANNEL_METAL_ROUGH, 2))
        {
            textures.metalRoughOrSpecular.rg = float2(
                channels[CHANNEL_METAL_ROUGH + 0],
                channels[CHANNEL_METAL_ROUGH + 1]);
        }
    }

    if (NtcTextureSetHasChannels(g_NtcMaterial, CHANNEL_NORMAL, 3))
    {
        textures.normal.rgb = float3(
            channels[CHANNEL_NORMAL + 0],
            channels[CHANNEL_NORMAL + 1],
            channels[CHANNEL_NORMAL + 2]);
    }

    if (NtcTextureSetHasChannels(g_NtcMaterial, CHANNEL_OCCLUSION, 1))
    {
        textures.occlusion.r = channels[CHANNEL_OCCLUSION];
    }
    
    if (NtcTextureSetHasChannels(g_NtcMaterial, CHANNEL_EMISSIVE, 3))
    {
        textures.emissive.rgb = float3(
            channels[CHANNEL_EMISSIVE + 0],
            channels[CHANNEL_EMISSIVE + 1],
            channels[CHANNEL_EMISSIVE + 2]);

        if (!linearizeColorsOnSample)
            textures.emissive.rgb = NtcSrgbColorSpace::Decode(textures.emissive.rgb);
    }
    else if (NtcTextureSetHasChannels(g_NtcMaterial, CHANNEL_EMISSIVE, 1))
    {
        float emissive = channels[CHANNEL_EMISSIVE];

        if (!linearizeColorsOnSample)
            emissive = NtcSrgbColorSpace::Decode(emissive);

        textures.emissive.rgb = emissive.rrr;
    }

    if (NtcTextureSetHasChannels(g_NtcMaterial, CHANNEL_TRANSMISSION, 1))
    {
        textures.transmission.r = channels[CHANNEL_TRANSMISSION];
    }

    return textures;
}
#endif


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
#if NETWORK_VERSION == NTC_NETWORK_UNKNOWN
    MaterialTextureSample textures = DefaultMaterialTextures();
#else
    MaterialTextureSample textures = SampleNtcMaterial(int2(i_position.xy), i_vtx.texCoord);
#endif

    // Force the MetalnessInRedChannel flag because it might not be set in the material constants
    // when inference on load is not used.
    MaterialConstants materialConstants = g_Material;
    materialConstants.flags |= MaterialFlags_MetalnessInRedChannel;
    materialConstants.flags |= MaterialFlags_UseOpacityTexture;

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
