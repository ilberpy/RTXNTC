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

// in FORWARD_SPACE_MATERIAL
#define FORWARD_BINDING_NTC_MATERIAL_CONSTANTS 4
#define FORWARD_BINDING_NTC_LATENTS_BUFFER 0
#define FORWARD_BINDING_NTC_WEIGHTS_BUFFER 1

// in FORWARD_SPACE_SHADING
#define FORWARD_BINDING_NTC_PASS_CONSTANTS 5
#define FORWARD_BINDING_STF_SAMPLER 1

struct NtcForwardShadingPassConstants
{
    uint frameIndex;
    uint stfFilterMode;
};
