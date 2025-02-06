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
    bctest
    DESCRIPTION "BC Compression Test"
    LANGUAGES CXX
)

find_package(CUDAToolkit REQUIRED)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF) 

add_executable(bctest)

set_target_properties(bctest PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY                "${NTC_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG          "${NTC_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_MINSIZEREL     "${NTC_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE        "${NTC_BINARY_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELWITHDEBINFO "${NTC_BINARY_DIR}"
)

target_link_libraries(bctest PRIVATE libntc stb argparse donut_app donut_engine)
target_link_libraries(bctest PRIVATE ntc-utils)

target_sources(bctest PRIVATE 
    BCTest.cpp
)

if (TARGET nvtt3)
    target_link_libraries(bctest PRIVATE nvtt3)
    target_compile_definitions(bctest PRIVATE NTC_WITH_NVTT=1)

    if (WIN32)
        # Copy the NVTT3 DLL to the output directory
        add_custom_command( TARGET bctest POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_RUNTIME_DLLS:bctest> $<TARGET_FILE_DIR:bctest>
            COMMAND_EXPAND_LISTS)
    endif()
else()
    target_compile_definitions(bctest PRIVATE NTC_WITH_NVTT=0)
endif()

if (DONUT_WITH_DX12)
    add_dependencies(bctest dx12-agility-sdk)
endif()
