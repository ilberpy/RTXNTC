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

namespace donut::app
{
    struct DeviceCreationParameters;
}

void SetNtcGraphicsDeviceParameters(
    donut::app::DeviceCreationParameters& deviceParams,
    nvrhi::GraphicsAPI graphicsApi,
    bool enableSharedMemory,
    char const* windowTitle);

bool IsDP4aSupported(nvrhi::IDevice* device);

bool IsFloat16Supported(nvrhi::IDevice* device);

bool IsDX12DeveloperModeEnabled();
