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

#include <ntc-utils/Semantics.h>
#include <algorithm>
#include <cstdint>

void GuessImageSemantics(std::string const& distinctName, int channels, ntc::ChannelFormat channelFormat,
    int imageIndex, bool &outIsSRGB, std::vector<SemanticBinding>& outSemantics)
{
    std::string lowercaseName = distinctName;
    std::transform(lowercaseName.begin(), lowercaseName.end(), lowercaseName.begin(), [](uint8_t ch) { return std::tolower(ch); });

    bool isSDR = channelFormat == ntc::ChannelFormat::UNORM8 || channelFormat == ntc::ChannelFormat::UNORM16;
    outIsSRGB = isSDR;

    if ((lowercaseName.find("diffuse") != std::string::npos ||
            lowercaseName.find("alb") != std::string::npos ||
            lowercaseName.find("color") != std::string::npos) && channels >= 3)
    {
        outSemantics.push_back({ SemanticLabel::Albedo, imageIndex, 0 });
        if (channels == 4 && isSDR) // Assume that HDR images do not have an alpha channel
            outSemantics.push_back({ SemanticLabel::AlphaMask, imageIndex, 3 });
    }

    if ((lowercaseName.find("normal") != std::string::npos ||
            lowercaseName.find("nrm") != std::string::npos) && channels >= 3)
    {
        outSemantics.push_back({ SemanticLabel::Normal, imageIndex, 0 });
        outIsSRGB = false;
    }
    else if ((lowercaseName.find("orm") != std::string::npos ||
                lowercaseName.find("arm") != std::string::npos) && channels >= 3) // "ORM" but not "nORMal"
    {
        outSemantics.push_back({ SemanticLabel::Occlusion, imageIndex, 0 });
        outSemantics.push_back({ SemanticLabel::Roughness, imageIndex, 1 });
        outSemantics.push_back({ SemanticLabel::Metalness, imageIndex, 2 });
        outIsSRGB = false;
    }
    else if ((lowercaseName.find("rma") != std::string::npos) && channels >= 3)
    {
        outSemantics.push_back({ SemanticLabel::Roughness, imageIndex, 0 });
        outSemantics.push_back({ SemanticLabel::Metalness, imageIndex, 1 });
        outSemantics.push_back({ SemanticLabel::Occlusion, imageIndex, 2 });
        outIsSRGB = false;
    }

    if (lowercaseName.find("occlusion") != std::string::npos ||
        lowercaseName.find("ambient") != std::string::npos ||
        lowercaseName.find("ao") != std::string::npos)
    {
        outSemantics.push_back({ SemanticLabel::Occlusion, imageIndex, 0 });
        outIsSRGB = false;
    }

    if (lowercaseName.find("roughness") != std::string::npos)
    {
        outSemantics.push_back({ SemanticLabel::Roughness, imageIndex, 0 });
        outIsSRGB = false;
    }

    if (lowercaseName.find("metal") != std::string::npos) // metalness or metallic
    {
        outSemantics.push_back({ SemanticLabel::Metalness, imageIndex, 0 });
        outIsSRGB = false;
    }

    if (lowercaseName.find("mask") != std::string::npos)
    {
        outSemantics.push_back({ SemanticLabel::AlphaMask, imageIndex, 0 });
    }

    if (lowercaseName.find("emissive") != std::string::npos && channels >= 3)
    {
        outSemantics.push_back({ SemanticLabel::Emissive, imageIndex, 0 });
    }

    if (lowercaseName.find("disp") != std::string::npos)
    {
        outSemantics.push_back({ SemanticLabel::Displacement, imageIndex, 0 });
        outIsSRGB = false;
    }
}
