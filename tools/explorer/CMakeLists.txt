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
    ntc-explorer
    DESCRIPTION "Neural Texture Compression Explorer"
    LANGUAGES CXX
)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) 

add_executable(ntc-explorer WIN32)

set_target_properties(ntc-explorer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY                "${NTC_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG          "${NTC_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL     "${NTC_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE        "${NTC_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${NTC_BINARY_DIR}"
)

target_link_libraries(ntc-explorer PRIVATE CUDA::cudart_static libntc argparse donut_render donut_app donut_engine)
target_link_libraries(ntc-explorer PRIVATE ntc-utils)

target_sources(ntc-explorer PRIVATE 
    NtcExplorer.cpp
    FlatImageView.cpp
    FlatImageView.h
    FlatImageViewConstants.h
    ImGuiExtensions.cpp
    ImGuiExtensions.h
    ModelView.cpp
    ModelView.h
    ModelViewConstants.h
)

include(${CMAKE_SOURCE_DIR}/external/donut/compileshaders.cmake)

set(shader_sources
    FlatImageView.hlsl
    ModelView.hlsl)

set(shader_outputs
    FlatImageView_MainPS
    ModelView_MainPS
    ModelView_MainVS
    ModelView_OverlayPS)

set(shader_output_dir "${CMAKE_CURRENT_BINARY_DIR}/compiled_shaders")
target_include_directories(ntc-explorer PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")

set(libntc_include_directory "${CMAKE_SOURCE_DIR}/libraries/RTXNTC-Library/include")

donut_compile_shaders_all_platforms(
    TARGET ntc-explorer-shaders
    CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/Shaders.cfg
    OUTPUT_BASE ${shader_output_dir}
    OUTPUT_FORMAT HEADER
    SOURCES ${shader_sources}
    BYPRODUCTS_NO_EXT ${shader_outputs}
    INCLUDES ${libntc_include_directory}
    SHADERMAKE_OPTIONS "--hlsl2021"
)

add_dependencies(ntc-explorer ntc-explorer-shaders)

if (DONUT_WITH_DX12)
    add_dependencies(ntc-explorer dx12-agility-sdk)
endif()
