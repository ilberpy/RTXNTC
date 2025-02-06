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

#include "FlatImageViewConstants.h"
#include <libntc/shaders/ColorSpaces.hlsli>
#include <donut/shaders/binding_helpers.hlsli>

#ifdef __cplusplus
static const FlatImageViewConstants g_Const;
#else
VK_PUSH_CONSTANT ConstantBuffer<FlatImageViewConstants> g_Const : register(b0);
#endif

Texture2D t_LeftInput : register(t0);
Texture2D t_RightInput : register(t1);
RWBuffer<float4> u_PixelBuffer : register(u0);
SamplerState s_InputSampler : register(s0);

static const float c_CheckerboardSizePixels = 8.0;
static const float c_CheckerboardBrightColor = 0.35;
static const float c_CheckerboardDarkColor = 0.65;

float4 ConvertInput(float4 textureValue)
{
    if (countbits(g_Const.channelMask) == 1)
    {
        if ((g_Const.channelMask & 1) != 0) textureValue.rgb = textureValue.rrr; else
        if ((g_Const.channelMask & 2) != 0) textureValue.rgb = textureValue.ggg; else
        if ((g_Const.channelMask & 4) != 0) textureValue.rgb = textureValue.bbb; else
                                            textureValue.rgb = textureValue.aaa;
        textureValue.a = 1.0;
    }
    else
    {
        if ((g_Const.channelMask & 1) == 0) textureValue.r = 0.0;
        if ((g_Const.channelMask & 2) == 0) textureValue.g = 0.0;
        if ((g_Const.channelMask & 4) == 0) textureValue.b = 0.0;
        if ((g_Const.channelMask & 8) == 0) textureValue.a = 1.0;
    }
    return textureValue;
}

float Luminance(float3 color)
{
    return 0.2126 * color.r + 0.7152 * color.g + 0.0722 * color.b;
}

float3 ReinhardToneMapping(float3 color)
{
    float srcLuminance = Luminance(color);

    if (srcLuminance <= 0)
        return 0;

    const float whitePoint = 10.0; // Increasing this parameter makes the whites less crushed but more washed out
    const float whitePointInvSquared = 1.0 / (whitePoint * whitePoint);
    const float mappedLuminance = (srcLuminance * (1 + srcLuminance * whitePointInvSquared)) / (1 + srcLuminance);

    return color * (mappedLuminance / srcLuminance);
}

float GetRelativeDifference(float a, float b)
{
    if (a == 0 && b == 0)
        return 0;
    return abs(a - b) / max(abs(a), abs(b));
}

float4 MainPS(
    in float4 windowPos : SV_Position,
    in float2 quadUv : UV) : SV_Target0
{
    float2 relativePos = windowPos.xy - g_Const.viewCenter - g_Const.textureCenterOffset;
    int2 scaledTextureSize = g_Const.textureSize * g_Const.displayScale;
    float2 uv = (relativePos + floor(scaledTextureSize * 0.5)) / scaledTextureSize;

    float4 textureColor = 0;

    if (all(uv >= 0.0) && all(uv <= 1.0))
    {
        float4 leftValue = t_LeftInput.Sample(s_InputSampler, uv);
        float4 rightValue = t_RightInput.Sample(s_InputSampler, uv);
        
        if (all(g_Const.pixelPickPosition == int2(windowPos.xy)))
        {
            u_PixelBuffer[0] = leftValue;
            u_PixelBuffer[1] = rightValue;
        }

        leftValue = ConvertInput(leftValue);
        rightValue = ConvertInput(rightValue);
                
        switch(g_Const.displayMode)
        {
            case DisplayMode::LeftTexture:
                textureColor = leftValue;
                break;
            case DisplayMode::RightTexture:
                textureColor = rightValue;
                break;
            case DisplayMode::Difference:
                textureColor.rgb = abs(leftValue.rgb - rightValue.rgb);
                textureColor.a = leftValue.a;
                break;
            case DisplayMode::RelativeDifference:
                textureColor.r = GetRelativeDifference(leftValue.r, rightValue.r);
                textureColor.g = GetRelativeDifference(leftValue.g, rightValue.g);
                textureColor.b = GetRelativeDifference(leftValue.b, rightValue.b);
                textureColor.a = leftValue.a;
                break;
            case DisplayMode::SplitScreen:
                textureColor = windowPos.x < float(g_Const.splitPosition)
                    ? leftValue
                    : rightValue;
        }

        if (g_Const.isSRGB)
            textureColor.rgb = NtcSrgbColorSpace::Decode(textureColor.rgb);

        textureColor.rgb *= g_Const.colorScale;

        if (g_Const.applyToneMapping)
            textureColor.rgb = ReinhardToneMapping(textureColor.rgb);
    }

    const int2 checkerboardPos = int2(floor(relativePos / c_CheckerboardSizePixels));
    const float checkerboardColor = ((checkerboardPos.x + checkerboardPos.y) & 1)
        ? c_CheckerboardBrightColor
        : c_CheckerboardDarkColor;

    float3 finalColor = lerp(checkerboardColor.xxx, textureColor.rgb, saturate(textureColor.a));

    if (g_Const.pixelHighlightBottomRight.x > g_Const.pixelHighlightTopLeft.x)
    {
        const int top = g_Const.pixelHighlightTopLeft.y;
        const int left = g_Const.pixelHighlightTopLeft.x;
        const int right = g_Const.pixelHighlightBottomRight.x;
        const int bottom = g_Const.pixelHighlightBottomRight.y;
        const int x = int(windowPos.x);
        const int y = int(windowPos.y);

        if ((left <= x && x <= right && (y == top || y == bottom)) ||
            (top <= y && y <= bottom && (x == left || x == right)))
        {
            const int lineColor = (((x - left) >> 2) + ((y - top) >> 2)) & 1;
            finalColor.rgb = float(lineColor).xxx;
        }
    }

    const int distanceFromSplit = abs(int(windowPos.x) - g_Const.splitPosition);
    if (g_Const.displayMode == DisplayMode::SplitScreen && distanceFromSplit < 2)
    {
        finalColor = (distanceFromSplit == 0) ? 1.0 : 0.0; // Black bar with white in the middle
    }

    return float4(finalColor, 1.0);
}