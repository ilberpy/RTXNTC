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

#pragma once

#include <imgui.h>

namespace ImGui
{
// Toggle button works as a checkbox but is more compact. State is shown as color. Using a border is recommended.
bool ToggleButton(const char* label, bool* state, const ImVec2& size_arg = ImVec2(0.f, 0.f), ImGuiButtonFlags flags = 0);

// Version of ToggleButton that operates on a single bit from 'state'.
bool ToggleButtonFlags(const char* label, ImU32* state, ImU32 bit, const ImVec2& size_arg = ImVec2(0.f, 0.f), ImGuiButtonFlags flags = 0);

// Shows a question mark and a popup window on hover.
void TooltipMarker(const char* desc);

}
