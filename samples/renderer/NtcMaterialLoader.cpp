/*
 * SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#include "NtcMaterialLoader.h"
#include "NtcMaterial.h"
#include <ntc-utils/GraphicsDecompressionPass.h>
#include <ntc-utils/GraphicsBlockCompressionPass.h>
#include <ntc-utils/DeviceUtils.h>

#include <donut/core/log.h>
#include <donut/core/string_utils.h>
#include <donut/engine/Scene.h>

#include <sstream>
#include <fstream>

using namespace donut;
namespace fs = std::filesystem;

bool NtcMaterialLoader::Init(bool enableCoopVecInt8, bool enableCoopVecFP8, nvrhi::ITexture* dummyTexture)
{
    ntc::ContextParameters contextParams;
    contextParams.cudaDevice = ntc::DisableCudaDevice;
    contextParams.graphicsApi = m_device->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D12
        ? ntc::GraphicsAPI::D3D12
        : ntc::GraphicsAPI::Vulkan;

    bool const osSupportsCoopVec = (contextParams.graphicsApi == ntc::GraphicsAPI::D3D12)
        ? IsDX12DeveloperModeEnabled()
        : true;
    
    contextParams.d3d12Device = m_device->getNativeObject(nvrhi::ObjectTypes::D3D12_Device);
    contextParams.vkInstance = m_device->getNativeObject(nvrhi::ObjectTypes::VK_Instance);
    contextParams.vkPhysicalDevice = m_device->getNativeObject(nvrhi::ObjectTypes::VK_PhysicalDevice);
    contextParams.vkDevice = m_device->getNativeObject(nvrhi::ObjectTypes::VK_Device);
    contextParams.graphicsDeviceSupportsDP4a = IsDP4aSupported(m_device);
    contextParams.graphicsDeviceSupportsFloat16 = IsFloat16Supported(m_device);
    contextParams.enableCooperativeVectorInt8 = osSupportsCoopVec && enableCoopVecInt8;
    contextParams.enableCooperativeVectorFP8 = osSupportsCoopVec && enableCoopVecFP8;

    ntc::Status ntcStatus = ntc::CreateContext(m_ntcContext.ptr(), contextParams);
    if (ntcStatus != ntc::Status::Ok && ntcStatus != ntc::Status::CudaUnavailable)
    {
        log::error("Failed to create an NTC context, code = %s: ",
            ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
        return false;
    }

    m_coopVecInt8 = m_ntcContext->IsCooperativeVectorInt8Supported();
    m_coopVecFP8 = m_ntcContext->IsCooperativeVectorFP8Supported();

    m_dummyTexture = std::make_shared<engine::LoadedTexture>();
    m_dummyTexture->texture = dummyTexture;

    m_graphicsDecompressionPass = std::make_shared<GraphicsDecompressionPass>(m_device,
        /* descriptorTableSize = */ 8 * NTC_MAX_MIPS);
    if (!m_graphicsDecompressionPass->Init())
        return false;

    m_graphicsBlockCompressionPass = std::make_shared<GraphicsBlockCompressionPass>(m_device,
        /* useAccelerationBuffer = */ false, /* maxConstantBufferVersions = */ 128);
    if (!m_graphicsBlockCompressionPass->Init())
        return false;

    m_commandList = m_device->createCommandList(nvrhi::CommandListParameters().setEnableImmediateExecution(false));

    return true;
}

static bool LoadMaterialFile(fs::path const& ntcFileName, NtcMaterial& material, ntc::IContext* ntcContext,
    ntc::FileStreamWrapper& ntcFile, ntc::TextureSetMetadataWrapper& textureSetMetadata)
{
    if (material.name.empty())
        material.name = "Material";

    ntc::Status ntcStatus = ntcContext->OpenFile(ntcFileName.generic_string().c_str(), false, ntcFile.ptr());
    if (ntcStatus == ntc::Status::FileUnavailable)
    {
        log::warning("Material file '%s' does not exist.", ntcFileName.generic_string().c_str());
        return false;
    }
    else if (ntcStatus != ntc::Status::Ok)
    {
        log::warning("Cannot open '%s', error code = %s: %s", ntcFileName.generic_string().c_str(),
            ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
        return false;
    }

    ntcStatus = ntcContext->CreateTextureSetMetadataFromStream(ntcFile, textureSetMetadata.ptr());
    if (ntcStatus != ntc::Status::Ok)
    {
        log::warning("Cannot load metadata for '%s', error code = %s: %s", ntcFileName.generic_string().c_str(),
            ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
        return false;
    }

    material.networkVersion = textureSetMetadata->GetNetworkVersion();

    return true;
}

bool NtcMaterialLoader::TranscodeMaterial(ntc::IStream* ntcFile, ntc::StreamRange streamRange,
    ntc::ITextureSetMetadata* textureSetMetadata, NtcMaterial& material, nvrhi::ICommandList* commandList,
    bool enableBlockCompression, bool onlyAlphaMask)
{
    struct TextureVersions
    {
        ntc::ITextureMetadata const* metadata = nullptr;
        ntc::BlockCompressedFormat bcFormat = ntc::BlockCompressedFormat::None;
        nvrhi::TextureHandle color;
        nvrhi::TextureHandle blocks;
        nvrhi::TextureHandle compressed;
    };

    std::vector<TextureVersions> materialTextures;
    int textureCount = textureSetMetadata->GetTextureCount();

    // Create TextureVersions structures for every input texture
    materialTextures.resize(textureCount);

    // Per our fixed material channel mapping to NTC channels, the base color texture is in channels 0-3,
    // and alpha mask is the .a component in that texture.
    int const alphaMaskChannel = 3;

    // If we only need to create the alpha mask texture, see if the material actually needs an alpha mask
    // and if the NTC texture set has an alpha mask channel.

    int alphaMaskTextureIndex = -1;
    if (onlyAlphaMask)
    {
        if (material.domain != donut::engine::MaterialDomain::AlphaTested &&
            material.domain != donut::engine::MaterialDomain::TransmissiveAlphaTested)
            return true;

        for (int textureIndex = 0; textureIndex < textureCount; ++textureIndex)
        {
            ntc::ITextureMetadata const* textureMetadata = textureSetMetadata->GetTexture(textureIndex);
            int firstChannel, numChannels;
            textureMetadata->GetChannels(firstChannel, numChannels);
            if (firstChannel <= alphaMaskChannel && firstChannel + numChannels > alphaMaskChannel)
            {
                alphaMaskTextureIndex = textureIndex;
                break;
            }
        }

        if (alphaMaskTextureIndex < 0)
            return true;
    }

    // Phase 1 - Create textures (color, block, BCn) and write descriptors for NTC decompression

    ntc::TextureSetDesc const& textureSetDesc = textureSetMetadata->GetDesc();
    for (int textureIndex = 0; textureIndex < textureCount; ++textureIndex)
    {
        if (onlyAlphaMask && textureIndex != alphaMaskTextureIndex)
            continue;

        ntc::ITextureMetadata const* textureMetadata = textureSetMetadata->GetTexture(textureIndex);
        bool const sRGB = textureMetadata->GetRgbColorSpace() == ntc::ColorSpace::sRGB;
        std::string const textureName = textureMetadata->GetName();
        std::string const materialTextureName = material.name + ":" + textureMetadata->GetName();

        TextureVersions& textureVersions = materialTextures[textureIndex];
        textureVersions.metadata = textureMetadata;
        textureVersions.bcFormat = onlyAlphaMask
            ? ntc::BlockCompressedFormat::BC4
            : textureMetadata->GetBlockCompressedFormat();

        // Create the color texture

        nvrhi::TextureDesc colorTextureDesc = nvrhi::TextureDesc()
            .setDimension(nvrhi::TextureDimension::Texture2D)
            .setWidth(textureSetDesc.width)
            .setHeight(textureSetDesc.height)
            .setMipLevels(textureSetDesc.mips)
            .setFormat(onlyAlphaMask
                ? nvrhi::Format::R8_UNORM
                : sRGB
                    ? nvrhi::Format::SRGBA8_UNORM
                    : nvrhi::Format::RGBA8_UNORM)
            .setDebugName(materialTextureName)
            .setIsUAV(true)
            .setIsTypeless(true)
            .setInitialState(nvrhi::ResourceStates::ShaderResource)
            .setKeepInitialState(true);

        textureVersions.color = m_device->createTexture(colorTextureDesc);
        if (!textureVersions.color)
            return false;

        bool compressThisTexture = enableBlockCompression && textureVersions.bcFormat != ntc::BlockCompressedFormat::None;

        if (compressThisTexture)
        {
            // Create the BCn texture

            nvrhi::TextureDesc compressedTextureDesc = nvrhi::TextureDesc()
                .setDimension(nvrhi::TextureDimension::Texture2D)
                .setWidth(textureSetDesc.width)
                .setHeight(textureSetDesc.height)
                .setMipLevels(textureSetDesc.mips)
                .setDebugName(materialTextureName)
                .setInitialState(nvrhi::ResourceStates::ShaderResource)
                .setKeepInitialState(true);

            switch(textureVersions.bcFormat)
            {
                case ntc::BlockCompressedFormat::BC1:
                    compressedTextureDesc.setFormat(sRGB ? nvrhi::Format::BC1_UNORM_SRGB : nvrhi::Format::BC1_UNORM);
                    break;
                case ntc::BlockCompressedFormat::BC2:
                    compressedTextureDesc.setFormat(sRGB ? nvrhi::Format::BC2_UNORM_SRGB : nvrhi::Format::BC2_UNORM);
                    break;
                case ntc::BlockCompressedFormat::BC3:
                    compressedTextureDesc.setFormat(sRGB ? nvrhi::Format::BC3_UNORM_SRGB : nvrhi::Format::BC3_UNORM);
                    break;
                case ntc::BlockCompressedFormat::BC4:
                    compressedTextureDesc.setFormat(nvrhi::Format::BC4_UNORM);
                    break;
                case ntc::BlockCompressedFormat::BC5:
                    compressedTextureDesc.setFormat(nvrhi::Format::BC5_UNORM);
                    break;
                case ntc::BlockCompressedFormat::BC6:
                    compressedTextureDesc.setFormat(nvrhi::Format::BC6H_UFLOAT);
                    break;
                case ntc::BlockCompressedFormat::BC7:
                    compressedTextureDesc.setFormat(sRGB ? nvrhi::Format::BC7_UNORM_SRGB : nvrhi::Format::BC7_UNORM);
                    break;
                default:
                    log::warning("Material '%s' texture '%s': pixel format %s is recognized as block compressed, "
                        "but it's not BC1-7.", material.name.c_str(), textureName.c_str(),
                        ntc::BlockCompressedFormatToString(textureVersions.bcFormat));
                    compressThisTexture = false;
            }

            if (compressThisTexture)
            {
                textureVersions.compressed = m_device->createTexture(compressedTextureDesc);
                if (!textureVersions.compressed)
                    return false;
            }
        }

        if (compressThisTexture)
        {
            // Create the block texture

            bool const isSmallBlock =
                (textureVersions.bcFormat == ntc::BlockCompressedFormat::BC1) ||
                (textureVersions.bcFormat == ntc::BlockCompressedFormat::BC4);

            nvrhi::TextureDesc blockTextureDesc = nvrhi::TextureDesc()
                .setDimension(nvrhi::TextureDimension::Texture2D)
                .setWidth((textureSetDesc.width + 3) / 4)
                .setHeight((textureSetDesc.height + 3) / 4)
                .setFormat(isSmallBlock ? nvrhi::Format::RG32_UINT : nvrhi::Format::RGBA32_UINT)
                .setDebugName(materialTextureName)
                .setIsUAV(true)
                .setInitialState(nvrhi::ResourceStates::UnorderedAccess)
                .setKeepInitialState(true);

            textureVersions.blocks = m_device->createTexture(blockTextureDesc);
            if (!textureVersions.blocks)
                return false;
        }
        
        // Write descriptors for all mips of the color texture
        for (int mipLevel = 0; mipLevel < textureSetDesc.mips; ++mipLevel)
        {
            // Descriptors for a single mip of all textures need to be in continuous slots
            // because the NTC decompression pass expects that layout.
            int descriptorIndex = mipLevel * textureCount + textureIndex;

            nvrhi::BindingSetItem descriptor = nvrhi::BindingSetItem::Texture_UAV(
                descriptorIndex,
                textureVersions.color,
                onlyAlphaMask
                    ? nvrhi::Format::R8_UNORM
                    : nvrhi::Format::RGBA8_UNORM, // Always use non-sRGB formats so that we can create a UAV
                nvrhi::TextureSubresourceSet().setBaseMipLevel(mipLevel));

            m_graphicsDecompressionPass->WriteDescriptor(descriptor);
        }

        // Transition the texture to the UAV state because NVRHI won't do that when resources are accessed
        // through a descriptor table. Note that there is no need to transition it back to SRV after decompression
        // because the next operations are using regular binding sets. There is also no need for commitBarriers()
        // because that's called by the decompression dispatch call.
        commandList->setTextureState(textureVersions.color, nvrhi::AllSubresources,
            nvrhi::ResourceStates::UnorderedAccess);
        
        // Create a LoadedTexture object to attach the texture to the material
        std::shared_ptr<engine::LoadedTexture> loadedTexture = std::make_shared<engine::LoadedTexture>();
        loadedTexture->texture = compressThisTexture ? textureVersions.compressed : textureVersions.color;

        // Count the final texture size in the material's memory consumption metric
        size_t const textureMemorySize = m_device->getTextureMemoryRequirements(loadedTexture->texture).size;
        material.transcodedMemorySize += textureMemorySize;
        
        // Determine which slot the texture goes into based on its name
        if (textureName == "BaseColor" || textureName == "DiffuseColor")
        {
            if (onlyAlphaMask)
                material.opacityTexture = loadedTexture;
            else
                material.baseOrDiffuseTexture = loadedTexture;
        }
        else if (textureName == "MetallicRoughness" || textureName == "SpecularGlossiness")
            material.metalRoughOrSpecularTexture = loadedTexture;
        else if (textureName == "Normal")
            material.normalTexture = loadedTexture;
        else if (textureName == "Occlusion")
            material.occlusionTexture = loadedTexture;
        else if (textureName == "Emissive")
            material.emissiveTexture = loadedTexture;
        else if (textureName == "Transmission")
            material.transmissionTexture = loadedTexture;
        else
        {
            log::warning("Material '%s' includes unrecognized texture '%s', skipping.",
                material.name.c_str(), textureName.c_str());
        }
    }

    // Submit the texture transitions performed above via setTextureState(...) to the command list.
    // This is not really necessary because the next call to commandList->setComputeState(...) will do it,
    // but let's be explicit.
    commandList->commitBarriers();

    // Phase 2 - Run NTC decompression

    if (material.ntcLatentsBuffer)
    {
        // If the data buffer has been previously created for Inference On Sample, use that.
        m_graphicsDecompressionPass->SetInputBuffer(material.ntcLatentsBuffer);
    }
    else
        m_graphicsDecompressionPass->SetInputData(commandList, ntcFile, streamRange);

    for (int mipLevel = 0; mipLevel < textureSetDesc.mips; ++mipLevel)
    {
        // alphaDesc is used when we're decompressing only the alpha channel, in which case it describes
        // which channel to process and where to put the result.
        ntc::OutputTextureDesc alphaDesc;
        alphaDesc.firstChannel = alphaMaskChannel;
        alphaDesc.numChannels = 1;
        alphaDesc.descriptorIndex = alphaMaskTextureIndex;

        // Obtain the description of the decompression pass from LibNTC.
        // The description includes the shader code, weights, and constants.
        ntc::MakeDecompressionComputePassParameters decompressionParams;
        decompressionParams.textureSetMetadata = textureSetMetadata;
        decompressionParams.latentStreamRange = streamRange;
        decompressionParams.mipLevel = mipLevel;
        decompressionParams.firstOutputDescriptorIndex = mipLevel * textureCount;
        decompressionParams.pOutputTextures = &alphaDesc;
        decompressionParams.numOutputTextures = onlyAlphaMask ? 1 : 0;
        decompressionParams.enableFP8 = true;
        ntc::ComputePassDesc decompressionPass;
        ntc::Status ntcStatus = m_ntcContext->MakeDecompressionComputePass(decompressionParams, &decompressionPass);
        if (ntcStatus != ntc::Status::Ok)
        {
            log::warning("Failed to make a decompression pass for material '%s' mip %d, error code = %s: %s",
                material.name.c_str(), mipLevel, ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
            return false;
        }

        // Execute the compute pass to decompress the texture.
        // Note: ExecuteComputePass is application code (not LibNTC) and it caches PSOs based on shader code pointers.
        m_graphicsDecompressionPass->ExecuteComputePass(commandList, decompressionPass);
    }

    // Phase 3 - Compress all mips of the color textures into BCn, where necessary

    assert(textureCount == int(materialTextures.size()));
    for (int textureIndex = 0; textureIndex < textureCount; ++textureIndex)
    {
        TextureVersions const& textureVersions = materialTextures[textureIndex];
        if (!textureVersions.compressed)
            continue;

        ntc::ITextureMetadata const* textureMetadata = textureSetMetadata->GetTexture(textureIndex);

        float const alphaThreshold = 1.f / 255.f;
        
        for (int mipLevel = 0; mipLevel < textureSetDesc.mips; ++mipLevel)
        {
            int const mipWidth = std::max(textureSetDesc.width >> mipLevel, 1);
            int const mipHeight = std::max(textureSetDesc.height >> mipLevel, 1);

            // Obtain the description of the BC compression pass from LibNTC.
            ntc::MakeBlockCompressionComputePassParameters compressionParams;
            compressionParams.srcRect.width = mipWidth;
            compressionParams.srcRect.height = mipHeight;
            compressionParams.dstFormat = textureVersions.bcFormat;
            compressionParams.alphaThreshold = alphaThreshold;
            compressionParams.texture = textureMetadata;
            compressionParams.quality = textureMetadata->GetBlockCompressionQuality();
            ntc::ComputePassDesc compressionPass;
            ntc::Status ntcStatus = m_ntcContext->MakeBlockCompressionComputePass(compressionParams, &compressionPass);

            if (ntcStatus != ntc::Status::Ok)
            {
                log::warning("Failed to make a block compression pass for material '%s', error code = %s: %s",
                    material.name.c_str(), ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
                return false;
            }

            // Execute the compute pass to compress the texture.
            // Note: ExecuteComputePass is application code (not LibNTC) and it caches PSOs based on shader code pointers.
            if (!m_graphicsBlockCompressionPass->ExecuteComputePass(commandList, compressionPass,
                textureVersions.color, onlyAlphaMask ? nvrhi::Format::R8_UNORM : nvrhi::Format::RGBA8_UNORM,
                mipLevel, textureVersions.blocks, 0, nullptr))
                return false;
            
            int const mipWidthBlocks = (mipWidth + 3) / 4;
            int const mipHeightBlocks = (mipHeight + 3) / 4;

            commandList->copyTexture(textureVersions.compressed, nvrhi::TextureSlice().setMipLevel(mipLevel),
                textureVersions.blocks, nvrhi::TextureSlice().setWidth(mipWidthBlocks).setHeight(mipHeightBlocks));
        }
    }
    
    // We use custom texture packing that puts metalness and roughness into one NTC "texture"
    // with Metalness in R channel and Roughness in G channel.
    // Note: Only set this flag when Inference on Load is active, otherwise we get rendering corruption
    // because reference materials store ORM in that order.
    material.metalnessInRedChannel = true;

    return true;
}

bool NtcMaterialLoader::PrepareMaterialForInferenceOnSample(ntc::IStream* ntcFile, ntc::StreamRange streamRange,
    ntc::ITextureSetMetadata* textureSetMetadata, NtcMaterial& material, nvrhi::ICommandList* commandList)
{
    ntc::InferenceWeightType weightType;
    if (m_coopVecFP8 && textureSetMetadata->IsInferenceWeightTypeSupported(ntc::InferenceWeightType::CoopVecFP8))
        weightType = ntc::InferenceWeightType::CoopVecFP8;
    else if (m_coopVecInt8 && textureSetMetadata->IsInferenceWeightTypeSupported(ntc::InferenceWeightType::CoopVecInt8))
        weightType = ntc::InferenceWeightType::CoopVecInt8;
    else
        weightType = ntc::InferenceWeightType::GenericInt8;

    ntc::InferenceData inferenceData;
    ntc::Status ntcStatus = m_ntcContext->MakeInferenceData(textureSetMetadata, streamRange,
        weightType, &inferenceData);

    if (ntcStatus != ntc::Status::Ok)
    {
        log::warning("Failed to make inference data for material '%s', error code = %s: %s",
            material.name.c_str(), ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
        return false;
    }

    void const* weightData = nullptr;
    size_t weightSize = 0;
    ntcStatus = textureSetMetadata->GetInferenceWeights(weightType, &weightData, &weightSize);

    if (ntcStatus != ntc::Status::Ok)
    {
        log::warning("Failed to get inference weights for material '%s', error code = %s: %s",
            material.name.c_str(), ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
        return false;
    }

    nvrhi::BufferDesc constantBufferDesc = nvrhi::BufferDesc()
        .setByteSize(sizeof(inferenceData.constants))
        .setIsConstantBuffer(true)
        .setInitialState(nvrhi::ResourceStates::ConstantBuffer)
        .setKeepInitialState(true)
        .setDebugName(material.name + " constants");
    material.ntcConstantBuffer = m_device->createBuffer(constantBufferDesc);
    if (!material.ntcConstantBuffer)
        return false;

    nvrhi::BufferDesc weightBufferDesc = nvrhi::BufferDesc()
        .setByteSize(weightSize)
        .setCanHaveRawViews(true)
        .setInitialState(nvrhi::ResourceStates::ShaderResource)
        .setKeepInitialState(true)
        .setDebugName(material.name + " weights");
    material.ntcWeightsBuffer = m_device->createBuffer(weightBufferDesc);
    if (!material.ntcWeightsBuffer)
        return false;

    nvrhi::BufferDesc latentBufferDesc = nvrhi::BufferDesc()
        .setByteSize(streamRange.size)
        .setCanHaveRawViews(true)
        .setInitialState(nvrhi::ResourceStates::ShaderResource)
        .setKeepInitialState(true)
        .setDebugName(material.name + " latents");
    material.ntcLatentsBuffer = m_device->createBuffer(latentBufferDesc);
    if (!material.ntcLatentsBuffer)
        return false;

    std::vector<uint8_t> latentData;
    latentData.resize(streamRange.size);

    ntcFile->Seek(streamRange.offset);
    if (!ntcFile->Read(latentData.data(), latentData.size()))
    {
        log::warning("Failed to read latents for material '%s'", material.name.c_str());
        return false;
    }

    commandList->writeBuffer(material.ntcLatentsBuffer, latentData.data(), latentData.size());
    commandList->writeBuffer(material.ntcWeightsBuffer, weightData, weightSize);
    commandList->writeBuffer(material.ntcConstantBuffer, &inferenceData.constants,
        sizeof(inferenceData.constants));

    int const textureCount = textureSetMetadata->GetTextureCount();
    for (int textureIndex = 0; textureIndex < textureCount; ++textureIndex)
    {
        ntc::ITextureMetadata const* textureMetadata = textureSetMetadata->GetTexture(textureIndex);
        std::string const textureName = textureMetadata->GetName();

        if (textureName == "BaseColor" || textureName == "DiffuseColor")
            material.baseOrDiffuseTexture = m_dummyTexture;
        else if (textureName == "MetallicRoughness" || textureName == "SpecularGlossiness")
            material.metalRoughOrSpecularTexture = m_dummyTexture;
        else if (textureName == "Normal")
            material.normalTexture = m_dummyTexture;
        else if (textureName == "Occlusion")
            material.occlusionTexture = m_dummyTexture;
        else if (textureName == "Emissive")
            material.emissiveTexture = m_dummyTexture;
        else if (textureName == "Transmission")
            material.transmissionTexture = m_dummyTexture;
    }

    material.ntcMemorySize =
        m_device->getBufferMemoryRequirements(material.ntcConstantBuffer).size + 
        m_device->getBufferMemoryRequirements(material.ntcWeightsBuffer).size + 
        m_device->getBufferMemoryRequirements(material.ntcLatentsBuffer).size;
    
    material.weightType = int(weightType);

    return true;
}

bool NtcMaterialLoader::LoadMaterialsForScene(donut::engine::Scene& scene, std::filesystem::path const& materialDir, 
    bool enableInferenceOnLoad, bool enableBlockCompression, bool enableInferenceOnSample)
{
    using namespace std::chrono;
    time_point start = steady_clock::now();

    uint64_t totalFileSize = 0;
    uint64_t totalPixels = 0;
    int materialCount = 0;

    std::vector<std::shared_ptr<engine::Material>> materials;
    for (std::shared_ptr<engine::Material> const& material : scene.GetSceneGraph()->GetMaterials())
        materials.push_back(material);

    std::unordered_map<std::string, std::vector<std::string>> materialToNtcMappings; // modelFileName -> [ntcFileName]
    std::unordered_map<fs::path, std::shared_ptr<NtcMaterial>> ntcMaterialCache; // ntcFileName -> NtcMaterial

    for (std::shared_ptr<engine::Material> const& material : materials)
    {
        NtcMaterial* ntcMaterial = static_cast<NtcMaterial*>(material.get());

        fs::path modelFileName = material->modelFileName;
        fs::path currentMaterialDir = materialDir.empty() ? modelFileName.parent_path() : materialDir;

        auto mappingIterator = materialToNtcMappings.find(material->modelFileName);
        if (mappingIterator == materialToNtcMappings.end())
        {
            std::vector<std::string> mapping;

            fs::path mappingFileName = currentMaterialDir / (modelFileName.stem().generic_string() + ".ntc-materials.txt");
            std::ifstream mappingFile(mappingFileName.generic_string());
            if (mappingFile)
            {
                std::string line;
                while (std::getline(mappingFile, line))
                {
                    donut::string_utils::trim(line);
                    mapping.push_back(line);
                }
            }

            auto [newIterator, inserted] = materialToNtcMappings.insert_or_assign(material->modelFileName, mapping);
            mappingIterator = newIterator;
        }

        std::vector<std::string> const& materialMapping = mappingIterator->second;
        fs::path ntcFileName;
        if (materialMapping.size() > material->materialIndexInModel)
        {
            std::string const& mappingEntry = materialMapping[material->materialIndexInModel];
            if (!mappingEntry.empty() && mappingEntry != "*")
            {
                ntcFileName = currentMaterialDir / mappingEntry;
            }
            else
            {
                // No NTC file specified in the mapping, skip.
                continue;
            }
        }
        else
        {
            ntcFileName = currentMaterialDir / (material->name + ".ntc");
        }

        auto cacheIterator = ntcMaterialCache.find(ntcFileName);
        if (cacheIterator != ntcMaterialCache.end())
        {
            auto previouslyLoadedMaterial = cacheIterator->second;

            // Copy over all the properties that we touch when decoding NTC materials,
            // but not the entire material: some flags or parameters might be different.
            ntcMaterial->ntcConstantBuffer = previouslyLoadedMaterial->ntcConstantBuffer;
            ntcMaterial->ntcWeightsBuffer = previouslyLoadedMaterial->ntcWeightsBuffer;
            ntcMaterial->ntcLatentsBuffer = previouslyLoadedMaterial->ntcLatentsBuffer;
            ntcMaterial->networkVersion = previouslyLoadedMaterial->networkVersion;
            ntcMaterial->baseOrDiffuseTexture = previouslyLoadedMaterial->baseOrDiffuseTexture;
            ntcMaterial->metalRoughOrSpecularTexture = previouslyLoadedMaterial->metalRoughOrSpecularTexture;
            ntcMaterial->normalTexture = previouslyLoadedMaterial->normalTexture;
            ntcMaterial->emissiveTexture = previouslyLoadedMaterial->emissiveTexture;
            ntcMaterial->occlusionTexture = previouslyLoadedMaterial->occlusionTexture;
            ntcMaterial->transmissionTexture = previouslyLoadedMaterial->transmissionTexture;
            ntcMaterial->metalnessInRedChannel = previouslyLoadedMaterial->metalnessInRedChannel;

            continue;
        }

        ntc::FileStreamWrapper ntcFile(m_ntcContext);
        ntc::TextureSetMetadataWrapper textureSetMetadata(m_ntcContext);

        fs::path modelPath = material->modelFileName;

        if (!LoadMaterialFile(ntcFileName, *ntcMaterial, m_ntcContext, ntcFile, textureSetMetadata))
            continue;

        // Obtain the stream range for latents covering all mip levels of the material.
        ntc::StreamRange streamRange;
        ntc::Status ntcStatus = textureSetMetadata->GetStreamRangeForLatents(0,
            textureSetMetadata->GetDesc().mips, streamRange);
        if (ntcStatus != ntc::Status::Ok)
        {
            log::warning("Cannot process material '%s', call to GetStreamRangeForLatents failed, error code = %s: %s",
                ntcMaterial->name.c_str(), ntc::StatusToString(ntcStatus), ntc::GetLastErrorMessage());
        }

        m_commandList->open();
        bool loadedSuccessfully = true;

        // Load the material data for Inference On Sample first, so that the data buffer can be reused for On Load.
        if (enableInferenceOnSample)
        {
            loadedSuccessfully = PrepareMaterialForInferenceOnSample(ntcFile, streamRange, textureSetMetadata,
                *ntcMaterial, m_commandList);
        }

        // Transcode the material into raw color data or BCn (Inference On Load).
        // When Inference on Load is disabled, we still go through the materials and extract alpha mask channels,
        // encoding them into BC4 when allowed. They are used for the depth pre-pass (or any-hit shaders
        // in a path tracing renderer).
        if (loadedSuccessfully)
        {
            loadedSuccessfully = TranscodeMaterial(ntcFile, streamRange, textureSetMetadata,
                *ntcMaterial, m_commandList, enableBlockCompression, !enableInferenceOnLoad);
        }
        
        m_commandList->close();

        if (loadedSuccessfully)
        {
            m_device->executeCommandList(m_commandList);
            m_device->waitForIdle();
            m_device->runGarbageCollection();
        }

        auto const& textureSetDesc = textureSetMetadata->GetDesc();
        totalFileSize += ntcFile->Size();
        totalPixels += (textureSetDesc.width * textureSetDesc.height * 4) / 3;
        ++materialCount;
    }

    time_point end = steady_clock::now();
    int64_t durationMs = duration_cast<milliseconds>(end - start).count();
    
    log::info("%d materials loaded in %lli ms - that's %.2f Mpix from %.2f MB", materialCount, durationMs,
        double(totalPixels) * 1e-6, double(totalFileSize) * 0x1p-20);

    return true;
}
