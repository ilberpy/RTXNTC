# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

add_library(nvtt3 SHARED IMPORTED GLOBAL)

set_target_properties(nvtt3 PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${NVTT3_INCLUDE_DIR}")

if (WIN32)
    set_target_properties(nvtt3 PROPERTIES
        IMPORTED_LOCATION "${NVTT3_RUNTIME_LIBRARY}"
        IMPORTED_IMPLIB "${NVTT3_LIBRARY}")
else()
    set_target_properties(nvtt3 PROPERTIES
        IMPORTED_LOCATION "${NVTT3_RUNTIME_LIBRARY}")
endif()
