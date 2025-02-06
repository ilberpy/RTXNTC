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

#pragma once

#include <ntc-utils/Manifest.h>

struct SemanticBinding
{
    SemanticLabel label = SemanticLabel(0);
    int imageIndex = 0;
    int firstChannel = 0; // Number of channels is defined by the label
};

void GuessImageSemantics(std::string const& distinctName, int channels, ntc::ChannelFormat channelFormat,
    int imageIndex, bool &outIsSRGB, std::vector<SemanticBinding>& outSemantics);
