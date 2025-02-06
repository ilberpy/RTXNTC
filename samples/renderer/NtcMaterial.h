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

#include <donut/engine/SceneGraph.h>

struct NtcMaterial : public donut::engine::Material
{
    nvrhi::BufferHandle ntcConstantBuffer;
    nvrhi::BufferHandle ntcWeightsBuffer;
    nvrhi::BufferHandle ntcLatentsBuffer;
    int networkVersion = 0;
    int weightType = 0;
    size_t transcodedMemorySize = 0;
    size_t ntcMemorySize = 0;
};

class NtcSceneTypeFactory : public donut::engine::SceneTypeFactory
{
public:
    std::shared_ptr<donut::engine::Material> CreateMaterial() override;
};
