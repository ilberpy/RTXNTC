# SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

project(
    ntc-cli
    DESCRIPTION "Neural Texture Compression Command Line Interface"
    LANGUAGES CXX
)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) 

add_executable(ntc-cli)

set_target_properties(ntc-cli PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY                "${NTC_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG          "${NTC_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL     "${NTC_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE        "${NTC_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${NTC_BINARY_DIR}"
)

target_link_libraries(ntc-cli PRIVATE CUDA::cudart_static libntc stb tinyexr argparse donut_app donut_engine lodepng)
target_link_libraries(ntc-cli PRIVATE ntc-utils)

target_sources(ntc-cli PRIVATE 
    NtcCommandLine.cpp
    GraphicsPasses.cpp
    GraphicsPasses.h
    Utils.cpp
    Utils.h
)

if (DONUT_WITH_DX12)
    add_dependencies(ntc-cli dx12-agility-sdk)
endif()
