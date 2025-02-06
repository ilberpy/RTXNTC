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

#ifndef MODEL_VIEW_CONSTANTS_H
#define MODEL_VIEW_CONSTANTS_H

#include <donut/shaders/light_cb.h>
#include <donut/shaders/view_cb.h>

struct ModelViewConstants
{
    PlanarViewConstants view;
    LightConstants light;

    float3 skyColor;
    float mipLevel;
    float3 groundColor;
    int decompressedTextureOffset;

    int enableSplitScreen;
    int splitPosition;
    int convertFromSrgbMask;

    int albedoTexture;
    int albedoChannel;
    int alphaTexture;
    int alphaChannel;
    int emissiveTexture;
    int emissiveChannel;
    int metalnessTexture;
    int metalnessChannel;
    int normalTexture;
    int normalChannel;
    int occlusionTexture;
    int occlusionChannel;
    int roughnessTexture;
    int roughnessChannel;
};

struct OverlayConstants
{
    int splitPosition;
};

#endif