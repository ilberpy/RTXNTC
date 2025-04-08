/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <libntc/ntc.h>
#include <nvrhi/nvrhi.h>
#include <filesystem>
#include <unordered_map>

#include "feedbackmanager/include/FeedbackManager.h"

struct NtcMaterial;
class GraphicsDecompressionPass;
class GraphicsBlockCompressionPass;

namespace donut::engine
{
    struct LoadedTexture;
    class Scene;
}

namespace ntc
{
    class IContext;
    class IStream;
    class ITextureSetMetadata;
}

class NtcMaterialLoader
{
public:
    NtcMaterialLoader(nvrhi::IDevice* device)
        : m_device(device)
    { }
    
    bool Init(bool enableCoopVecInt8, bool enableCoopVecFP8, nvrhi::ITexture* dummyTexture);

    bool IsCooperativeVectorInt8Supported() const { return m_coopVecInt8; }
    
    bool IsCooperativeVectorFP8Supported() const { return m_coopVecFP8; }

    bool LoadMaterialsForScene(donut::engine::Scene& scene, std::filesystem::path const& materialDir, 
        bool enableInferenceOnLoad, bool enableBlockCompression, bool enableInferenceOnSample,
        bool enableInferenceOnFeedback, std::shared_ptr<nvfeedback::FeedbackManager> feedbackManager);

    bool TranscodeTile(const NtcMaterial& material, const nvfeedback::FeedbackTextureTileInfo& tile, nvrhi::ICommandList* commandList,
        bool onlyAlphaMask, bool enableBlockCompression);

private:
    nvrhi::DeviceHandle m_device;
    nvrhi::CommandListHandle m_commandList;

    ntc::ContextWrapper m_ntcContext;

    bool m_coopVecInt8 = false;
    bool m_coopVecFP8 = false;

    std::shared_ptr<donut::engine::LoadedTexture> m_dummyTexture;

    std::shared_ptr<GraphicsDecompressionPass> m_graphicsDecompressionPass;
    std::shared_ptr<GraphicsBlockCompressionPass> m_graphicsBlockCompressionPass;

    // Textures for tile-based decompression and recompression
    std::vector<nvrhi::TextureHandle> m_texTileColorR8;
    std::vector<nvrhi::TextureHandle> m_texTileColorRGBA;
    std::vector<nvrhi::TextureHandle> m_texTileColorSRGBA;
    nvrhi::TextureHandle m_texTileBlocksRG;
    nvrhi::TextureHandle m_texTileBlocksRGBA;

    bool TranscodeMaterial(ntc::IStream* ntcFile,
        ntc::ITextureSetMetadata* textureSetMetadata, NtcMaterial& material, nvrhi::ICommandList* commandList,
        bool enableBlockCompression, bool onlyAlphaMask);

    bool PrepareMaterialForInferenceOnSample(ntc::IStream* ntcFile,
        ntc::ITextureSetMetadata* textureSetMetadata, NtcMaterial& material, nvrhi::ICommandList* commandList);

    bool PrepareFeedbackMaterial(std::shared_ptr<nvfeedback::FeedbackManager> feedbackManager,
        ntc::ITextureSetMetadata* textureSetMetadata, NtcMaterial& material, nvrhi::ICommandList* commandList, bool enableBlockCompression);
};