/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <libntc/ntc.h>
#include <unordered_map>

class GraphicsImageDifferencePass
{
public:
    GraphicsImageDifferencePass(nvrhi::IDevice* device, uint32_t maxQueries = 1)
        : m_device(device)
        , m_maxQueries(maxQueries)
        , m_mseValues(maxQueries * ChannelsPerQuery)
    { }

    bool Init();

    // Returns offset in the result buffer for a given query. This offset should be passed to 
    // MakeImageDifferenceComputePass as the 'outputOffset' parameter.
    uint32_t GetOffsetForQuery(uint32_t queryIndex) const;
    
    // Runs the image comparison pass described by 'computePass' for a pair of textures.
    // Note: ExecuteComputePass expects that the commandList is open, and leaves it open.
    // To get the comparison results, execute the command list, then call ReadResults and GetQueryResult.
    bool ExecuteComputePass(nvrhi::ICommandList* commandList, ntc::ComputePassDesc& computePass,
        nvrhi::ITexture* texture1, int mipLevel1, nvrhi::ITexture* texture2, int mipLevel2, uint32_t queryIndex);

    // Reads the query results from the GPU and stores them internally. This involves a WFI and a buffer mapping.
    bool ReadResults();

    // Returns the image comparison results for a given query.
    // Call ReadResults() once before this function.
    bool GetQueryResult(uint32_t queryIndex, float outPerChannelMSE[4], float* outOverallMSE, float* outOverallPSNR,
        int channels = 4, float maxSignalValue = 1.0f);

private:
    nvrhi::DeviceHandle m_device;
    std::unordered_map<const void*, nvrhi::ComputePipelineHandle> m_pipelines; // shader bytecode -> pipeline
    nvrhi::BindingLayoutHandle m_bindingLayout;
    nvrhi::BufferHandle m_outputBuffer;
    nvrhi::BufferHandle m_stagingBuffer;
    nvrhi::BufferHandle m_constantBuffer;
    uint32_t m_maxQueries = 0;
    std::vector<float> m_mseValues;
    bool m_resultsRead = false;

    static constexpr uint32_t ChannelsPerQuery = 4;
    static constexpr uint32_t BytesPerQuery = ChannelsPerQuery * 8;
};
