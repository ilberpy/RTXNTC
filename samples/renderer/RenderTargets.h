/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <nvrhi/nvrhi.h>
#include <donut/engine/FramebufferFactory.h>
#include <memory>

struct RenderTargets
{
    nvrhi::TextureHandle depth;
    nvrhi::TextureHandle color;
    nvrhi::TextureHandle resolvedColor;
    nvrhi::TextureHandle feedback1;
    nvrhi::TextureHandle feedback2;
    nvrhi::TextureHandle motionVectors;

    std::shared_ptr<donut::engine::FramebufferFactory> depthFramebufferFactory;
    std::shared_ptr<donut::engine::FramebufferFactory> framebufferFactory;
};
