/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include <ntc-utils/Misc.h>
#include "NvidiaSansRegularFont.h"

ntc::VersionInfo GetNtcSdkVersion()
{
    ntc::VersionInfo info{};
    // No version numbers for the SDK at this time, just branch/commit.
    info.branch = NTC_SDK_VERSION_BRANCH;
    info.commitHash = NTC_SDK_VERSION_HASH;
    return info;
}

void GetNvidiaSansFont(void const** pOutData, size_t* pOutSize)
{
    *pOutData = g_NvidiaSansRegularFontData;
    *pOutSize = g_NvidiaSansRegularFontSize;
}
