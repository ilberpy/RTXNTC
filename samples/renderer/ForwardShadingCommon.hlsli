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

#include "donut/shaders/forward_cb.h"
#include "donut/shaders/scene_material.hlsli"
#include "donut/shaders/forward_vertex.hlsli"
#include "donut/shaders/hash_based_rng.hlsli"
#include "donut/shaders/lighting.hlsli"
#include "donut/shaders/shadows.hlsli"
#include "donut/shaders/binding_helpers.hlsli"

float3 GetIncidentVector(float4 directionOrPosition, float3 surfacePos)
{
    if (directionOrPosition.w > 0)
        return normalize(surfacePos.xyz - directionOrPosition.xyz);
    else
        return directionOrPosition.xyz;
}

void EvaluateForwardShading(
    MaterialConstants materialConstants,
    MaterialSample surfaceMaterial,
    float3 surfaceWorldPos,
    bool isFrontFace,
    PlanarViewConstants view,
    ForwardShadingLightConstants forwardLight,
    out float4 o_color
#if TRANSMISSIVE_MATERIAL
    , out float4 o_backgroundBlendFactor
#endif
)
{
    if (!isFrontFace)
        surfaceMaterial.shadingNormal = -surfaceMaterial.shadingNormal;

#if ENABLE_ALPHA_TEST
    if (materialConstants.domain != MaterialDomain_Opaque)
        clip(surfaceMaterial.opacity - materialConstants.alphaCutoff);
#endif

    float3 viewIncident = GetIncidentVector(view.cameraDirectionOrPosition, surfaceWorldPos);

    float3 diffuseTerm = 0;
    float3 specularTerm = 0;

    [loop]
    for(uint nLight = 0; nLight < forwardLight.numLights; nLight++)
    {
        LightConstants light = forwardLight.lights[nLight];

        float3 diffuseRadiance, specularRadiance;
        ShadeSurface(light, surfaceMaterial, surfaceWorldPos, viewIncident, diffuseRadiance, specularRadiance);

        diffuseTerm += diffuseRadiance * light.color;
        specularTerm += specularRadiance * light.color;
    }

    float NdotV = saturate(-dot(surfaceMaterial.shadingNormal, viewIncident));

    {
        float3 ambientColor = lerp(forwardLight.ambientColorBottom.rgb, forwardLight.ambientColorTop.rgb,
            surfaceMaterial.shadingNormal.y * 0.5 + 0.5);

        diffuseTerm += ambientColor * surfaceMaterial.diffuseAlbedo * surfaceMaterial.occlusion;
        specularTerm += ambientColor * surfaceMaterial.specularF0 * surfaceMaterial.occlusion;
    }
    
#if TRANSMISSIVE_MATERIAL
    
    // See https://github.com/KhronosGroup/glTF/blob/master/extensions/2.0/Khronos/KHR_materials_transmission/README.md#transmission-btdf

    float dielectricFresnel = Schlick_Fresnel(0.04, NdotV);
    
    o_color.rgb = diffuseTerm * (1.0 - surfaceMaterial.transmission)
        + specularTerm
        + surfaceMaterial.emissiveColor;

    o_color.a = 1.0;

    float backgroundScalar = surfaceMaterial.transmission
        * (1.0 - dielectricFresnel);

    if (materialConstants.domain == MaterialDomain_TransmissiveAlphaBlended)
        backgroundScalar *= (1.0 - surfaceMaterial.opacity);
    
    o_backgroundBlendFactor.rgb = backgroundScalar;

    if (surfaceMaterial.hasMetalRoughParams)
    {
        // Only apply the base color and metalness parameters if the surface is using the metal-rough model.
        // Transmissive behavoir is undefined on specular-gloss materials by the glTF spec, but it is
        // possible that the application creates such material regardless.

        o_backgroundBlendFactor.rgb *= surfaceMaterial.baseColor * (1.0 - surfaceMaterial.metalness);
    }

    o_backgroundBlendFactor.a = 1.0;

#else // TRANSMISSIVE_MATERIAL

    o_color.rgb = diffuseTerm
        + specularTerm
        + surfaceMaterial.emissiveColor;

    if (materialConstants.domain == MaterialDomain_AlphaTested)
    {
        // Fix the fuzzy edges on alpha tested geometry.
        // See https://bgolus.medium.com/anti-aliased-alpha-test-the-esoteric-alpha-to-coverage-8b177335ae4f
        // Improved filtering quality by multiplying fwidth by sqrt(2).
        o_color.a = saturate((surfaceMaterial.opacity - materialConstants.alphaCutoff)
            / max(fwidth(surfaceMaterial.opacity) * 1.4142, 0.0001) + 0.5);
    }
    else
        o_color.a = surfaceMaterial.opacity;

#endif // TRANSMISSIVE_MATERIAL
}