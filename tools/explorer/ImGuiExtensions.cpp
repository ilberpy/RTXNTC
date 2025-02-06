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

#define IMGUI_DEFINE_MATH_OPERATORS
#include "ImGuiExtensions.h"
#include <imgui_internal.h>

namespace ImGui
{

bool ToggleButton(const char* label, bool* state, const ImVec2& size_arg, ImGuiButtonFlags flags)
{
    // The implementation is copied and adjusted from ImGui::ButtonEx(...)

    ImGuiWindow* window = GetCurrentWindow();
    if (window->SkipItems)
        return false;

    ImGuiContext& g = *GImGui;
    const ImGuiStyle& style = g.Style;
    const ImGuiID id = window->GetID(label);
    const ImVec2 label_size = CalcTextSize(label, NULL, true);

    ImVec2 pos = window->DC.CursorPos;
    if ((flags & ImGuiButtonFlags_AlignTextBaseLine) && style.FramePadding.y < window->DC.CurrLineTextBaseOffset)
        pos.y += window->DC.CurrLineTextBaseOffset - style.FramePadding.y;
    ImVec2 size = CalcItemSize(size_arg, label_size.x + style.FramePadding.x * 2.0f, label_size.y + style.FramePadding.y * 2.0f);

    const ImRect bb(pos, pos + size);
    ItemSize(size, style.FramePadding.y);
    if (!ItemAdd(bb, id))
        return false;
    
    bool hovered, held;
    bool pressed = ButtonBehavior(bb, id, &hovered, &held, flags);

    if (pressed)
        *state = !*state;

    // Render
    PushStyleVar(ImGuiStyleVar_FrameBorderSize, 1.f);
    const ImU32 col = GetColorU32(*state ? ImGuiCol_ButtonActive : hovered ? ImGuiCol_FrameBg : ImGuiCol_WindowBg);
    RenderNavHighlight(bb, id);
    RenderFrame(bb.Min, bb.Max, col, true, style.FrameRounding);
    PopStyleVar();
    
    if (g.LogEnabled)
        LogSetNextTextDecoration("[", "]");
    if (g.CurrentItemFlags & ImGuiItemFlags_Disabled)
        PushStyleColor(ImGuiCol_Text, style.Colors[ImGuiCol_TextDisabled]);
    RenderTextClipped(bb.Min + style.FramePadding, bb.Max - style.FramePadding, label, nullptr, &label_size, style.ButtonTextAlign, &bb);
    if (g.CurrentItemFlags & ImGuiItemFlags_Disabled)
        PopStyleColor();

    return pressed;
}

bool ToggleButtonFlags(const char* label, ImU32* state, ImU32 bit, const ImVec2& size_arg, ImGuiButtonFlags flags)
{
    bool boolState = ((*state) & bit) != 0;

    bool result = ToggleButton(label, &boolState, size_arg, flags);

    if (boolState)
        *state |= bit;
    else
        *state &= ~bit;

    return result;
}	

void TooltipMarker(const char* desc)
{
    ImGui::SameLine();
    ImGui::TextDisabled("(?)");
    if (ImGui::IsItemHovered())
    {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 32.f);
        ImGui::TextUnformatted(desc);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}
}
