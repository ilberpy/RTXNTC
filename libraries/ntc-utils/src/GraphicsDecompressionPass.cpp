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
}

bool GraphicsDecompressionPass::SetWeightsFromTextureSet(nvrhi::ICommandList* commandList,
    ntc::ITextureSetMetadata* textureSetMetadata, ntc::InferenceWeightType weightType)
{
    void const* uploadData = nullptr;
    size_t uploadSize = 0;
    size_t convertedSize = 0;
    textureSetMetadata->GetInferenceWeights(weightType, &uploadData, &uploadSize, &convertedSize);

    bool const uploadBufferNeeded = convertedSize != 0;

    // Create the weight upload buffer if it doesn't exist yet or if it is too small
    if (!m_weightUploadBuffer && uploadBufferNeeded ||
        m_weightUploadBuffer && m_weightUploadBuffer->getDesc().byteSize < uploadSize)
    {
        nvrhi::BufferDesc uploadBufferDesc;
        uploadBufferDesc
            .setByteSize(uploadSize)
            .setDebugName("DecompressionWeightsUpload")
            .setInitialState(nvrhi::ResourceStates::CopyDest)
            .setKeepInitialState(true);

        m_weightUploadBuffer = m_device->createBuffer(uploadBufferDesc);

        if (!m_weightUploadBuffer)
            return false;
    }

    size_t finalWeightBufferSize = convertedSize ? convertedSize : uploadSize;

    // Create the weight buffer if it doesn't exist yet or if it is too small
    if (!m_weightBuffer || m_weightBufferIsExternal || m_weightBuffer->getDesc().byteSize < finalWeightBufferSize)
    {
        nvrhi::BufferDesc weightBufferDesc;
        weightBufferDesc
            .setByteSize(finalWeightBufferSize)
            .setDebugName("DecompressionWeights")
            .setCanHaveRawViews(true)
            .setCanHaveUAVs(true)
            .setInitialState(nvrhi::ResourceStates::ShaderResource)
            .setKeepInitialState(true);

        m_weightBuffer = m_device->createBuffer(weightBufferDesc);
        m_weightBufferIsExternal = false;

        if (!m_weightBuffer)
            return false;
    }

    if (uploadBufferNeeded)
    {
        // Write the weight upload buffer
        commandList->writeBuffer(m_weightUploadBuffer, uploadData, uploadSize);

        // Place the barriers before layout conversion - which happens in LibNTC and bypasses NVRHI
        commandList->setBufferState(m_weightUploadBuffer, nvrhi::ResourceStates::ShaderResource);
        commandList->setBufferState(m_weightBuffer, nvrhi::ResourceStates::UnorderedAccess);
        commandList->commitBarriers();

        // Unwrap the command list and buffer objects from NVRHI
        bool const isVulkan = m_device->getGraphicsAPI() == nvrhi::GraphicsAPI::VULKAN;
        nvrhi::ObjectType const commandListType = isVulkan
            ? nvrhi::ObjectTypes::VK_CommandBuffer
            : nvrhi::ObjectTypes::D3D12_GraphicsCommandList;
        nvrhi::ObjectType const bufferType = isVulkan
            ? nvrhi::ObjectTypes::VK_Buffer
            : nvrhi::ObjectTypes::D3D12_Resource;

        void* nativeCommandList = commandList->getNativeObject(commandListType);
        void* nativeSrcBuffer = m_weightUploadBuffer->getNativeObject(bufferType);
        void* nativeDstBuffer = m_weightBuffer->getNativeObject(bufferType);

        // Convert the weight layout to CoopVec
        textureSetMetadata->ConvertInferenceWeights(weightType, nativeCommandList,
            nativeSrcBuffer, 0, nativeDstBuffer, 0);
    }
    else
    {
        // Write the weight buffer directly
        commandList->writeBuffer(m_weightBuffer, uploadData, uploadSize);
    }

    return true;
}

void GraphicsDecompressionPass::SetWeightBuffer(nvrhi::IBuffer* buffer)
{
    if (buffer == m_weightBuffer)
        return;
        
    m_weightBuffer = buffer;
    m_weightBufferIsExternal = true; // Prevent the buffer from being overwritten by a subsequent call to SetWeightsFromTextureSet
}

bool GraphicsDecompressionPass::ExecuteComputePass(nvrhi::ICommandList* commandList, ntc::ComputePassDesc& computePass)
{
    // Create the pipeline for this shader if it doesn't exist yet
    auto& pipeline = m_pipelines[computePass.computeShader];
    if (!pipeline)
    {
        nvrhi::ShaderHandle computeShader = m_device->createShader(nvrhi::ShaderDesc().setShaderType(nvrhi::ShaderType::Compute),
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

        if (!m_constantBuffer)
            return false;
    }

    nvrhi::BindingSetDesc bindingSetDesc;
    bindingSetDesc
        .addItem(nvrhi::BindingSetItem::ConstantBuffer(0, m_constantBuffer))
        .addItem(nvrhi::BindingSetItem::RawBuffer_SRV(1, m_inputBuffer))
        .addItem(nvrhi::BindingSetItem::RawBuffer_SRV(2, m_weightBuffer));
    nvrhi::BindingSetHandle bindingSet = m_bindingCache.GetOrCreateBindingSet(bindingSetDesc, m_bindingLayout);
    if (!bindingSet)
        return false;

    // Write the constant buffer
    commandList->writeBuffer(m_constantBuffer, computePass.constantBufferData, computePass.constantBufferSize);

    // Execute the compute shader for decompression
    nvrhi::ComputeState state;
    state.setPipeline(pipeline)
         .addBindingSet(bindingSet)
         .addBindingSet(m_descriptorTable);
    commandList->setComputeState(state);
    commandList->dispatch(computePass.dispatchWidth, computePass.dispatchHeight);

    return true;
}