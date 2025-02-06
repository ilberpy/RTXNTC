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

enum class DisplayMode
{
    LeftTexture,
    RightTexture,
    Difference,
    RelativeDifference,
    SplitScreen
};

struct FlatImageViewConstants
{
    float2 viewCenter;
    float2 textureCenterOffset;
    float2 textureSize;
    int2 pixelPickPosition;
    int2 pixelHighlightTopLeft;
    int2 pixelHighlightBottomRight;
    float displayScale;
    uint channelMask;
    int splitPosition;
    DisplayMode displayMode;
    float colorScale;
    uint applyToneMapping;
    uint isSRGB;
};
