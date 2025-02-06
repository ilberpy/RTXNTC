# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

find_package(PackageHandleStandardArgs)
include(FindPackageHandleStandardArgs)

set(NVTT3_SEARCH_PATHS "${NVTT3_SEARCH_PATH}")

if (WIN32)

    if (NOT NVTT3_SEARCH_PATHS)
        set (NVTT3_SEARCH_PATHS
            "$ENV{SystemDrive}/Program Files/NVIDIA Corporation/NVIDIA Texture Tools")
    endif()

    find_library(NVTT3_LIBRARY
        NAMES nvtt30205.lib
        PATHS ${NVTT3_SEARCH_PATHS}
        PATH_SUFFIXES lib/x64-v142
    )
        
    find_file(NVTT3_RUNTIME_LIBRARY
        NAMES nvtt30205.dll
        PATHS ${NVTT3_SEARCH_PATHS}
        PATH_SUFFIXES ""
    )

    find_path(NVTT3_INCLUDE_DIR
        NAMES nvtt/nvtt.h
        PATHS ${NVTT3_SEARCH_PATHS}
        PATH_SUFFIXES include
    )

    find_package_handle_standard_args(NVTT3
        REQUIRED_VARS
            NVTT3_INCLUDE_DIR
            NVTT3_LIBRARY
            NVTT3_RUNTIME_LIBRARY
    )

else()

    find_library(NVTT3_RUNTIME_LIBRARY
        NAMES libnvtt.so.30205
        PATHS ${NVTT3_SEARCH_PATHS}
    )

    find_path(NVTT3_INCLUDE_DIR
        NAMES nvtt/nvtt.h
        PATHS ${NVTT3_SEARCH_PATHS}
        PATH_SUFFIXES include
    )

    find_package_handle_standard_args(NVTT3
        REQUIRED_VARS
            NVTT3_INCLUDE_DIR
            NVTT3_RUNTIME_LIBRARY
    )
    
endif()
