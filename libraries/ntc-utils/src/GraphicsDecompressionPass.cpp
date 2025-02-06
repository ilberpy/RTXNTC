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

#include <ntc-utils/GraphicsDecompressionPass.h>

bool GraphicsDecompressionPass::Init()
{
    // Make sure the binding layout exists
    if (!m_bindingLayout)
    {
        nvrhi::VulkanBindingOffsets vulkanBindingOffsets;
        vulkanBindingOffsets
            .setConstantBufferOffset(0)
            .setSamplerOffset(0)
            .setShaderResourceOffset(0)
            .setUnorderedAccessViewOffset(0);

        nvrhi::BindingLayoutDesc layoutDesc;
        layoutDesc
            .setVisibility(nvrhi::ShaderType::Compute)
            .setBindingOffsets(vulkanBindingOffsets)
            .addItem(nvrhi::BindingLayoutItem::VolatileConstantBuffer(0))
            .addItem(nvrhi::BindingLayoutItem::RawBuffer_SRV(1))
            .addItem(nvrhi::BindingLayoutItem::RawBuffer_SRV(2));

        m_bindingLayout = m_device->createBindingLayout(layoutDesc);

        if (!m_bindingLayout)
            return false;
    }

    // Make sure the bindless layout exists
    if (!m_bindlessLayout)
    {
        nvrhi::BindlessLayoutDesc bindlessLayoutDesc;
        bindlessLayoutDesc
            .setVisibility(nvrhi::ShaderType::Compute)
            .setMaxCapacity(m_descriptorTableSize)
            .addRegisterSpace(nvrhi::BindingLayoutItem::Texture_UAV(0));

        m_bindlessLayout = m_device->createBindlessLayout(bindlessLayoutDesc);

        if (!m_bindlessLayout)
            return false;
    }

    // Make sure the descriptor table exists
    if (!m_descriptorTable)
    {
        m_descriptorTable = m_device->createDescriptorTable(m_bindlessLayout);
        if (!m_descriptorTable)
            return false;

        m_device->resizeDescriptorTable(m_descriptorTable, m_descriptorTableSize, false);
    }

    return true;
}

void GraphicsDecompressionPass::WriteDescriptor(nvrhi::BindingSetItem item)
{
    m_device->writeDescriptorTable(m_descriptorTable, item);
}

bool GraphicsDecompressionPass::SetInputData(nvrhi::ICommandList* commandList, ntc::IStream* inputStream,
    ntc::StreamRange range)
{
    if (range.size + 1 == 0)
    {
        range.size = inputStream->Size();
    }

    // Make sure that the decompression input and staging buffers exist and have sufficient size
    if (!m_inputBuffer || m_inputBufferIsExternal || m_inputBuffer->getDesc().byteSize < range.size)
    {
        nvrhi::BufferDesc inputBufferDesc;
        inputBufferDesc
            .setByteSize(range.size)
            .setDebugName("DecompressionInputData")
            .setCanHaveRawViews(true)
            .setInitialState(nvrhi::ResourceStates::ShaderResource)
            .setKeepInitialState(true);

        m_inputBuffer = m_device->createBuffer(inputBufferDesc);
        m_inputBufferIsExternal = false;
        m_bindingSet = nullptr;

        if (!m_inputBuffer)
            return false;
    }

    std::vector<uint8_t> latentsBuffer;
    latentsBuffer.resize(range.size);

    if (!inputStream->Seek(range.offset))
        return false;
        
    if (!inputStream->Read(latentsBuffer.data(), latentsBuffer.size()))
        return false;

    commandList->writeBuffer(m_inputBuffer, latentsBuffer.data(), latentsBuffer.size());

    return true;
}

void GraphicsDecompressionPass::SetInputBuffer(nvrhi::IBuffer* buffer)
{
    if (buffer == m_inputBuffer)
        return;
    
    m_inputBuffer = buffer;
    m_inputBufferIsExternal = true; // Prevent the buffer from being overwritten by a subsequent call to SetInputData
    m_bindingSet = nullptr;
}


bool GraphicsDecompressionPass::ExecuteComputePass(nvrhi::ICommandList* commandList, ntc::ComputePassDesc& computePass)
{
    // Create the pipeline for this shader if it doesn't exist yet
    auto& pipeline = m_pipelines[computePass.computeShader];
    if (!pipeline)
    {
        nvrhi::ShaderHandle computeShader = m_device->createShader(nvrhi::ShaderDesc(nvrhi::ShaderType::Compute),
                                                                   computePass.computeShader,
                                                                   computePass.computeShaderSize);

        nvrhi::ComputePipelineDesc pipelineDesc;
        pipelineDesc
            .setComputeShader(computeShader)
            .addBindingLayout(m_bindingLayout)
            .addBindingLayout(m_bindlessLayout);

        pipeline = m_device->createComputePipeline(pipelineDesc);

        if (!pipeline)
            return false;
    }

    // Create the constant buffer if it doesn't exist yet or if it is too small (which shouldn't happen currently)
    if (!m_constantBuffer || m_constantBuffer->getDesc().byteSize < computePass.constantBufferSize)
    {
        nvrhi::BufferDesc constantBufferDesc;
        constantBufferDesc
            .setByteSize(computePass.constantBufferSize)
            .setDebugName("DecompressionConstants")
            .setIsConstantBuffer(true)
            .setIsVolatile(true)
            .setMaxVersions(NTC_MAX_MIPS * NTC_MAX_CHANNELS);

        m_constantBuffer = m_device->createBuffer(constantBufferDesc);
        m_bindingSet = nullptr;

        if (!m_constantBuffer)
            return false;
    }

    // Create the weight buffer if it doesn't exist yet or if it is too small (which shouldn't happen currently)
    if (!m_weightBuffer || m_weightBuffer->getDesc().byteSize < computePass.weightBufferSize)
    {
        nvrhi::BufferDesc weightBufferDesc;
        weightBufferDesc
            .setByteSize(computePass.weightBufferSize)
            .setDebugName("DecompressionWeights")
            .setCanHaveRawViews(true)
            .setInitialState(nvrhi::ResourceStates::ShaderResource)
            .setKeepInitialState(true);

        m_weightBuffer = m_device->createBuffer(weightBufferDesc);
        m_bindingSet = nullptr;

        if (!m_weightBuffer)
            return false;
    }

    // Create the binding set if it doesn't exist yet
    if (!m_bindingSet)
    {
        nvrhi::BindingSetDesc bindingSetDesc;
        bindingSetDesc
            .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_constantBuffer))
            .addItem(nvrhi::BindingSetItem::RawBuffer_SRV(1, m_inputBuffer))
            .addItem(nvrhi::BindingSetItem::RawBuffer_SRV(2, m_weightBuffer));

        m_bindingSet = m_device->createBindingSet(bindingSetDesc, m_bindingLayout);

        if (!m_bindingSet)
            return false;
    }

    // Write the constant buffer
    commandList->writeBuffer(m_constantBuffer, computePass.constantBufferData, computePass.constantBufferSize);

    // Write the weight buffer
    commandList->writeBuffer(m_weightBuffer, computePass.weightBufferData, computePass.weightBufferSize);

    // Execute the compute shader for decompression
    nvrhi::ComputeState state;
    state.setPipeline(pipeline)
         .addBindingSet(m_bindingSet)
         .addBindingSet(m_descriptorTable);
    commandList->setComputeState(state);
    commandList->dispatch(computePass.dispatchWidth, computePass.dispatchHeight);

    return true;
}
