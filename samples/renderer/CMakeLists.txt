# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

project(
    ntc-renderer
    DESCRIPTION "Neural Texture Compression Scene Renderer"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) 

add_executable(ntc-renderer WIN32)

target_sources(ntc-renderer PRIVATE
    NtcMaterial.h
    NtcMaterialLoader.cpp
    NtcMaterialLoader.h
    NtcSceneRenderer.cpp
    NtcForwardShadingPass.cpp
    NtcForwardShadingPass.h
    NtcForwardShadingPassConstants.h
    Profiler.cpp
    Profiler.h
    RenderTargets.h
)

file(GLOB feedback_sources
    feedbackmanager/include/*.h
    feedbackmanager/src/*.h
    feedbackmanager/src/*.cpp
    ../../external/RTXTS-TTM/include/rtxts-ttm/*.h
    ../../external/RTXTS-TTM/src/*.h
    ../../external/RTXTS-TTM/src/*.cpp
)

if (DONUT_WITH_DX12)
    target_sources(ntc-renderer PRIVATE ${feedback_sources})
endif()

if (TARGET DLSS)
    target_sources(ntc-renderer PRIVATE 
        DLSS.cpp
        DLSS.h
    )
    if (DONUT_WITH_DX12)
        target_sources(ntc-renderer PRIVATE DLSS-DX12.cpp)
    endif()
    if (DONUT_WITH_VULKAN)
        target_sources(ntc-renderer PRIVATE DLSS-VK.cpp)
    endif()

    target_link_libraries(ntc-renderer PRIVATE DLSS)
    target_compile_definitions(ntc-renderer PRIVATE NTC_WITH_DLSS=1)

    add_custom_command(TARGET ntc-renderer POST_BUILD COMMAND
        cmake -E copy_if_different "${DLSS_SHARED_LIBRARY_PATH}" "${NTC_BINARY_DIR}/")
else()
    target_compile_definitions(ntc-renderer PRIVATE NTC_WITH_DLSS=0)
endif()

set_target_properties(ntc-renderer PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY                "${NTC_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG          "${NTC_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL     "${NTC_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE        "${NTC_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${NTC_BINARY_DIR}"
)

target_link_libraries(ntc-renderer PRIVATE libntc argparse donut_render donut_app donut_engine)
target_link_libraries(ntc-renderer PRIVATE ntc-utils)

include(${CMAKE_SOURCE_DIR}/external/donut/compileshaders.cmake)

set(shader_sources
    ForwardShadingCommon.hlsli
    LegacyForwardShadingPass.hlsl
    NtcForwardShadingPass_CoopVec.slang
    NtcForwardShadingPass.hlsl
    ForwardShadingPassFeedback.hlsl
)

set(shader_output_dir "${CMAKE_CURRENT_BINARY_DIR}/compiled_shaders")

set(shader_outputs
    NtcForwardShadingPass
    LegacyForwardShadingPass
    ForwardShadingPassFeedback)

set(shader_outputs_slang
    NtcForwardShadingPass_CoopVec)

set(libntc_include_directory "${CMAKE_SOURCE_DIR}/libraries/RTXNTC-Library/include")
set(libstf_include_directory "${CMAKE_SOURCE_DIR}/libraries/RTXTF-Library")
    
target_include_directories(ntc-renderer PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
target_include_directories(ntc-renderer PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/../../external/donut/shaders")
target_include_directories(ntc-renderer PRIVATE "${libstf_include_directory}")
target_include_directories(ntc-renderer PRIVATE "${CMAKE_SOURCE_DIR}/external/RTXTS-TTM/include")

add_custom_target(ntc-renderer-shaders
    DEPENDS ShaderMake
    SOURCES ${shader_sources})

set_source_files_properties(${shader_sources} PROPERTIES VS_TOOL_OVERRIDE "None") 

donut_compile_shaders_all_platforms(
    TARGET ntc-renderer-shaders
    CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/Shaders.cfg
    SOURCES ${shader_sources}
    BYPRODUCTS_NO_EXT ${shader_outputs}
    OUTPUT_FORMAT HEADER
    OUTPUT_BASE ${shader_output_dir}
    INCLUDES ${libntc_include_directory} ${libstf_include_directory}
    SHADERMAKE_OPTIONS "--hlsl2021"
    SHADERMAKE_OPTIONS_DXIL "--shaderModel 6_6"
    SHADERMAKE_OPTIONS_SPIRV "--vulkanVersion 1.2"
)

if (NTC_WITH_COOPVEC)
    set(SLANGC_PATH "${SLANG_COMPILER}")

    donut_compile_shaders_all_platforms(
        TARGET ntc-renderer-shaders
        CONFIG ${CMAKE_CURRENT_SOURCE_DIR}/SlangShaders.cfg
        SLANG
        SOURCES ${shader_sources}
        BYPRODUCTS_NO_EXT ${shader_outputs_slang}
        OUTPUT_FORMAT HEADER
        OUTPUT_BASE ${shader_output_dir}
        INCLUDES ${libntc_include_directory} ${libstf_include_directory}
        # Disable errorneous 'use of uninitialized variable' warnings
        SHADERMAKE_OPTIONS_DXIL "--shaderModel 6_8 -X -Wno-41016"
        # Disable errorneous 'use of uninitialized global variable' warnings
        SHADERMAKE_OPTIONS_SPIRV "-X \"-Wno-41017 -capability spvCooperativeVectorNV\""
    )
endif()

add_dependencies(ntc-renderer ntc-renderer-shaders)
