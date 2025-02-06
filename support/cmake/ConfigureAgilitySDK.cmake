# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

set(sdk_files
    D3D12Core.dll
    D3D12Core.pdb
    d3d12SDKLayers.dll
    d3d12SDKLayers.pdb)

set(target_path "${NTC_BINARY_DIR}/d3d12/")

if (NOT DONUT_D3D_AGILITY_SDK_VERSION OR NOT DONUT_D3D_AGILITY_SDK_LIBRARIES)
    message(SEND_ERROR "Agility SDK variables were not configured, please re-configure the project to download it.")
endif()

add_custom_target(dx12-agility-sdk)
set_property (TARGET dx12-agility-sdk PROPERTY FOLDER "Third-Party Libraries")

file(MAKE_DIRECTORY ${target_path})

foreach (filename ${DONUT_D3D_AGILITY_SDK_LIBRARIES})

    add_custom_command(TARGET dx12-agility-sdk POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different "${filename}" "${target_path}")

endforeach()
