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


#include <donut/render/DrawStrategy.h>
#include <donut/render/DepthPass.h>
#include <donut/render/SkyPass.h>
#include <donut/render/TemporalAntiAliasingPass.h>
#include <donut/app/ApplicationBase.h>
#include <donut/app/Camera.h>
#include <donut/app/imgui_renderer.h>
#include <donut/app/UserInterfaceUtils.h>
#include <donut/engine/ShaderFactory.h>
#include <donut/engine/CommonRenderPasses.h>
#include <donut/engine/TextureCache.h>
#include <donut/engine/Scene.h>
#include <donut/engine/FramebufferFactory.h>
#include <donut/engine/BindingCache.h>
#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>
#include <donut/core/math/math.h>
#include <nvrhi/utils.h>
#include <ntc-utils/DeviceUtils.h>
#include <ntc-utils/Misc.h>
#include <argparse.h>
#include <sstream>
#include <chrono>
#include <STFDefinitions.h>

#include "NtcMaterialLoader.h"
#include "NtcMaterial.h"
#include "NtcForwardShadingPass.h"
#include "Profiler.h"
#include "RenderTargets.h"
#if NTC_WITH_DLSS
#include "DLSS.h"
#endif

namespace fs = std::filesystem;

using namespace donut;

static const char* g_ApplicationName = "NTC Scene Renderer";

struct
{
    std::string scenePath;
    const char* materialDir = nullptr;
    bool debug = false;
    bool useVulkan = false;
    bool useDX12 = false;
    bool referenceMaterials = false;
    bool blockCompression = true;
    bool inferenceOnLoad = true;
    bool inferenceOnSample = true;
    bool enableCoopVec = true;
    bool enableCoopVecInt8 = true;
    bool enableCoopVecFP8 = true;
    bool enableDLSS = true;
    int adapterIndex = -1;
} g_options;

bool ProcessCommandLine(int argc, const char** argv)
{
    struct argparse_option options[] = {
        OPT_HELP(),
#if NTC_WITH_VULKAN
        OPT_BOOLEAN(0, "vk", &g_options.useVulkan, "Use Vulkan API"),
#endif
#if NTC_WITH_DX12
        OPT_BOOLEAN(0, "dx12", &g_options.useDX12, "Use DX12 API"),
#endif
        OPT_BOOLEAN(0, "debug", &g_options.debug, "Enable graphics debug runtime"),
        OPT_BOOLEAN(0, "referenceMaterials", &g_options.referenceMaterials, "Load materials from regular image files instead of NTC"),
        OPT_BOOLEAN(0, "blockCompression", &g_options.blockCompression, "Enable transcoding to BCn (default on, use --no-blockCompression)"),
        OPT_BOOLEAN(0, "inferenceOnLoad", &g_options.inferenceOnLoad, "Enable inference on load (default on, use --no-inferenceOnLoad)"),
        OPT_BOOLEAN(0, "inferenceOnSample", &g_options.inferenceOnSample, "Enable inference on sample (default on, use --no-inferenceOnSample)"),
        OPT_BOOLEAN(0, "coopVec", &g_options.enableCoopVec, "Enable all CoopVec extensions (default on, use --no-coopVec)"),
        OPT_BOOLEAN(0, "coopVecFP8", &g_options.enableCoopVecFP8, "Enable CoopVec extensions for FP8 math (default on, use --no-coopVecFP8)"),
        OPT_BOOLEAN(0, "coopVecInt8", &g_options.enableCoopVecInt8, "Enable CoopVec extensions for Int8 math (default on, use --no-coopVecInt8)"),
        OPT_BOOLEAN(0, "dlss", &g_options.enableDLSS, "Enable DLSS (default on, use --no-dlss)"),
        OPT_INTEGER(0, "adapter", &g_options.adapterIndex, "Index of the graphics adapter to use (use ntc-cli.exe --dx12|vk --listAdapters to find out)"),
        OPT_STRING(0, "materialDir", &g_options.materialDir, "Subdirectory near the scene file where NTC materials are located"),
        OPT_END()
    };

    static const char* usages[] = {
        "ntc-renderer.exe [options...] <path/to/scene.gltf>",
        nullptr
    };

    // Copy argv[] pointers into a temporary array, because argparse overwrites those,
    // and later DLSS cannot find the path to the executable, at least on Linux.
    const char** argvCopy = (const char**)alloca(sizeof(void*) * argc);
    for (int i = 0; i < argc; ++i)
        argvCopy[i] = argv[i];

    struct argparse argparse {};
    argparse_init(&argparse, options, usages, ARGPARSE_USE_MESSAGE_BUFFER | ARGPARSE_NEVER_EXIT);
    argparse_describe(&argparse, nullptr, "\nScene renderer using NTC materials.");
    int argparse_result = argparse_parse(&argparse, argc, argvCopy);
    if (argparse_result < 0)
    {
        if (argparse.messages)
        {
            bool isError = argparse_result != ARGPARSE_HELP;
#ifdef _WIN32
            MessageBoxA(NULL, argparse.messages, g_ApplicationName, MB_OK | (isError ? MB_ICONERROR : 0));
#else
            fprintf(isError ? stderr : stdout, "%s\n", argparse.messages);
#endif
        }
        argparse_cleanup(&argparse);
        return false;
    }

    if (argparse.out[0])
        g_options.scenePath = argparse.out[0];
        
    argparse_cleanup(&argparse);

    if (g_options.useDX12 && g_options.useVulkan)
    {
        log::error("Options --vk and --dx12 cannot be used at the same time.");
        return false;
    }

#if NTC_WITH_DX12 && NTC_WITH_VULKAN
    if (!g_options.useDX12 && !g_options.useVulkan)
    {
        // When both DX12 and Vulkan are supported, prefer Vulkan.
        // Vulkan API for Cooperative Vector inference is more stable than the DX12 one.
        g_options.useVulkan = true;
    }
#elif NTC_WITH_DX12 && !NTC_WITH_VULKAN
    g_options.useDX12 = true;
    g_options.useVulkan = false;
#elif !NTC_WITH_DX12 && NTC_WITH_VULKAN
    g_options.useDX12 = false;
    g_options.useVulkan = true;
#endif

    if (!g_options.enableCoopVec)
    {
        g_options.enableCoopVecInt8 = false;
        g_options.enableCoopVecFP8 = false;
    }

    if (g_options.scenePath.empty())
    {
        char const* defaultModelRelativePath = "assets/models/FlightHelmet/FlightHelmet.gltf";
        fs::path sdkRoot = app::GetDirectoryWithExecutable().parent_path().parent_path();
        fs::path defaultModel = sdkRoot / defaultModelRelativePath;

        if (fs::exists(defaultModel))
        {
            g_options.scenePath = defaultModel.generic_string();
        }
        else
        {
            log::error("Cannot find the default SDK model file '%s'. Please provide a path to a GLTF model "
                "or a JSON scene description file on the command line.", defaultModelRelativePath);
            return false;
        }
    }
    else if (!fs::exists(g_options.scenePath))
    {
        log::error("The specified scene file '%s' does not exist.", g_options.scenePath.c_str());
        return false;
    }

    if (g_options.referenceMaterials)
    {
        g_options.inferenceOnLoad = false;
        g_options.inferenceOnSample = false;
    }
    else if (!g_options.inferenceOnLoad && !g_options.inferenceOnSample)
    {
        log::error("The options --no-inferenceOnLoad and --no-inferenceOnSample cannot be used together.");
        return false;
    }

    return true;
}

// A texture cache that refuses to load any textures from files.
class DummyTextureCache : public engine::TextureCache
{
public:
    using engine::TextureCache::TextureCache;

    std::shared_ptr<engine::LoadedTexture> LoadTextureFromFile(const std::filesystem::path& path, bool sRGB,
        engine::CommonRenderPasses* passes, nvrhi::ICommandList* commandList) override
    {
        return nullptr;
    }

    std::shared_ptr<engine::LoadedTexture> LoadTextureFromFileDeferred(const std::filesystem::path& path, bool sRGB) override
    {
        return nullptr;
    }

#ifdef DONUT_WITH_TASKFLOW
    std::shared_ptr<engine::LoadedTexture> LoadTextureFromFileAsync(const std::filesystem::path& path, bool sRGB,
        tf::Executor& executor) override
    {
        return nullptr;
    }
#endif
};

enum class AntiAliasingMode
{
    Off,
    TAA
#if NTC_WITH_DLSS
    , DLSS
#endif
};

class NtcSceneRenderer : public app::ImGui_Renderer
{
private:
    nvrhi::CommandListHandle m_commandList;
    
    RenderTargets m_renderTargets;

    
    std::unique_ptr<render::DepthPass> m_depthPass;
    std::unique_ptr<NtcForwardShadingPass> m_ntcForwardShadingPass;
    
    std::shared_ptr<engine::CommonRenderPasses> m_commonPasses;
    std::shared_ptr<engine::TextureCache> m_textureCache;
    std::shared_ptr<engine::ShaderFactory> m_shaderFactory;
    std::unique_ptr<engine::Scene> m_scene;
    std::unique_ptr<engine::BindingCache> m_bindingCache;
    std::shared_ptr<engine::DirectionalLight> m_light;
    std::shared_ptr<render::SkyPass> m_skyPass;
    std::unique_ptr<render::TemporalAntiAliasingPass> m_taaPass;
    AveragingTimerQuery m_prePassTimer;
    AveragingTimerQuery m_renderPassTimer;
    std::unique_ptr<NtcMaterialLoader> m_materialLoader;
#if NTC_WITH_DLSS
    std::unique_ptr<DLSS> m_DLSS;
#endif

    app::SwitchableCamera m_camera;
    engine::PlanarView m_view;
    engine::PlanarView m_previousView;
    AntiAliasingMode m_aaMode = AntiAliasingMode::TAA;
    std::shared_ptr<app::RegisteredFont> m_primaryFont = nullptr;
    std::shared_ptr<app::RegisteredFont> m_largerFont = nullptr;
    bool m_previousFrameValid = false;
    bool m_enableVsync = false;
    bool m_useSTF = true;
    int m_stfFilterMode = STF_FILTER_TYPE_CUBIC;
    bool m_inferenceOnSample = true;
    std::string m_screenshotFileName;
    bool m_screenshotWithUI = true;
    bool m_useDepthPrepass = true;

    size_t m_ntcTextureMemorySize = 0;
    size_t m_transcodedTextureMemorySize = 0;
    size_t m_referenceTextureMemorySize = 0;

public:
    NtcSceneRenderer(app::DeviceManager *deviceManager)
        : ImGui_Renderer(deviceManager)
        , m_prePassTimer(deviceManager->GetDevice())
        , m_renderPassTimer(deviceManager->GetDevice())
    {
        m_shaderFactory = std::make_shared<engine::ShaderFactory>(GetDevice(), nullptr, fs::path());
        m_commonPasses = std::make_shared<engine::CommonRenderPasses>(GetDevice(), m_shaderFactory);
        m_bindingCache = std::make_unique<engine::BindingCache>(GetDevice());
        m_materialLoader = std::make_unique<NtcMaterialLoader>(GetDevice());

#if NTC_WITH_DLSS
    if (g_options.enableDLSS)
    {
    #if NTC_WITH_DX12
        if (GetDevice()->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D12)
            m_DLSS = DLSS::CreateDX12(GetDevice(), *m_shaderFactory);
    #endif
    #if NTC_WITH_VULKAN
        if (GetDevice()->getGraphicsAPI() == nvrhi::GraphicsAPI::VULKAN)
            m_DLSS = DLSS::CreateVK(GetDevice(), *m_shaderFactory);
    #endif
        if (m_DLSS->IsSupported())
            m_aaMode = AntiAliasingMode::DLSS;
    }
#endif

        ImGui::GetIO().IniFilename = nullptr;
    }

    bool LoadScene(std::shared_ptr<vfs::IFileSystem> fs, const std::filesystem::path& sceneFileName) 
    {
        auto stf = std::make_shared<NtcSceneTypeFactory>();
        m_scene = std::make_unique<engine::Scene>(GetDevice(), *m_shaderFactory, fs, m_textureCache, nullptr, stf);

        if (!m_scene->Load(sceneFileName))
            return false;
        
        if (!g_options.referenceMaterials)
        {
            fs::path const materialDir = g_options.materialDir ? fs::path(g_options.materialDir) : fs::path();

            if (!m_materialLoader->LoadMaterialsForScene(*m_scene, materialDir,
                g_options.inferenceOnLoad, g_options.blockCompression, g_options.inferenceOnSample))
            {
                return false;
            }
        }

        m_scene->FinishedLoading(GetFrameIndex());

        m_textureCache->ProcessRenderingThreadCommands(*m_commonPasses, 0.f);
        m_textureCache->LoadingFinished();

        // Calculate the texture memory metrics
        m_referenceTextureMemorySize = 0;
        m_ntcTextureMemorySize = 0;
        m_transcodedTextureMemorySize = 0;
        if (g_options.referenceMaterials)
        {
            for (auto it = m_textureCache->begin(); it != m_textureCache->end(); ++it)
            {
                m_referenceTextureMemorySize += GetDevice()->getTextureMemoryRequirements(it->second->texture).size;
            }
        }
        else
        {
            for (std::shared_ptr<engine::Material> const& material : m_scene->GetSceneGraph()->GetMaterials())
            {
                NtcMaterial const& ntcMaterial = static_cast<NtcMaterial const&>(*material);
                m_ntcTextureMemorySize += ntcMaterial.ntcMemorySize;
                m_transcodedTextureMemorySize += ntcMaterial.transcodedMemorySize;
            }
        }

        auto const& sceneCameras = m_scene->GetSceneGraph()->GetCameras();
        if (!sceneCameras.empty())
            m_camera.SwitchToSceneCamera(sceneCameras[0]);

        return true;
    }

    void AddDirectionalLight()
    {
        m_light = std::make_shared<engine::DirectionalLight>();
        auto sceneGraph = m_scene->GetSceneGraph();
        sceneGraph->AttachLeafNode(sceneGraph->GetRootNode(), m_light);
        
        m_light->SetDirection(dm::double3(-1.0, -1.0, -1.0));
        m_light->angularSize = 1.f;
        m_light->irradiance = 5.f;

        sceneGraph->Refresh(GetFrameIndex());
    }

    void SetDefaultCamera()
    {
        auto sceneBoundingBox = m_scene->GetSceneGraph()->GetRootNode()->GetGlobalBoundingBox();
        float const diagonalLength = length(sceneBoundingBox.diagonal());
        
        app::ThirdPersonCamera& thirdPersonCamera = m_camera.GetThirdPersonCamera();
        thirdPersonCamera.SetTargetPosition(sceneBoundingBox.center());
        thirdPersonCamera.SetDistance(diagonalLength);
        thirdPersonCamera.SetRotation(dm::radians(-135.f), dm::radians(20.f));
        thirdPersonCamera.SetMoveSpeed(3.f);
        thirdPersonCamera.SetRotateSpeed(0.002f);
        
        app::FirstPersonCamera& firstPersonCamera = m_camera.GetFirstPersonCamera();
        firstPersonCamera.SetMoveSpeed(diagonalLength * 0.1f);
        firstPersonCamera.SetRotateSpeed(0.002f);
    }
    
    bool Init()
    {
        if (!m_materialLoader->Init(g_options.enableCoopVecInt8, g_options.enableCoopVecFP8,
            m_commonPasses->m_BlackTexture))
            return false;

        if (!ImGui_Renderer::Init(m_shaderFactory))
            return false;

        auto nativeFS = std::make_shared<vfs::NativeFileSystem>();

        if (g_options.referenceMaterials)
            m_textureCache = std::make_shared<engine::TextureCache>(GetDevice(), nativeFS, nullptr);
        else
            m_textureCache = std::make_shared<DummyTextureCache>(GetDevice(), nativeFS, nullptr);

        m_commandList = GetDevice()->createCommandList(nvrhi::CommandListParameters()
            .setEnableImmediateExecution(false)); // Disable immediate execution in case we abandon command lists

        if (!LoadScene(nativeFS, g_options.scenePath))
            return false;

        AddDirectionalLight();
        SetDefaultCamera();
        
        m_ntcForwardShadingPass = std::make_unique<NtcForwardShadingPass>(GetDevice(),
            m_shaderFactory, m_commonPasses);

        if (!m_ntcForwardShadingPass->Init())
            return false;
        
        m_depthPass = std::make_unique<render::DepthPass>(GetDevice(), m_commonPasses);
        render::DepthPass::CreateParameters depthParams;
        depthParams.numConstantBufferVersions = 128;
        m_depthPass->Init(*m_shaderFactory, depthParams);

        m_inferenceOnSample = g_options.inferenceOnSample;

        void const* pFontData;
        size_t fontSize;
        GetNvidiaSansFont(&pFontData, &fontSize);
        m_primaryFont = CreateFontFromMemoryCompressed(pFontData, fontSize, 16.f);
        m_largerFont = CreateFontFromMemoryCompressed(pFontData, fontSize, 22.f);

        return true;
    }
    
    void CreateRenderTargets(int width, int height)
    {
        auto textureDesc = nvrhi::TextureDesc()
            .setDimension(nvrhi::TextureDimension::Texture2D)
            .setWidth(width)
            .setHeight(height)
            .setClearValue(nvrhi::Color(0.f))
            .setIsRenderTarget(true)
            .setKeepInitialState(true);

        m_renderTargets.depth = GetDevice()->createTexture(textureDesc
            .setDebugName("Depth")
            .setFormat(nvrhi::Format::D32)
            .setInitialState(nvrhi::ResourceStates::DepthWrite));

        m_renderTargets.color = GetDevice()->createTexture(textureDesc
            .setDebugName("Color")
            .setFormat(nvrhi::Format::RGBA16_FLOAT)
            .setInitialState(nvrhi::ResourceStates::RenderTarget));

        m_renderTargets.resolvedColor = GetDevice()->createTexture(textureDesc
            .setDebugName("ResolvedColor")
            .setFormat(nvrhi::Format::RGBA16_FLOAT)
            .setIsRenderTarget(false)
            .setIsUAV(true)
            .setUseClearValue(false)
            .setInitialState(nvrhi::ResourceStates::UnorderedAccess));

        m_renderTargets.feedback1 = GetDevice()->createTexture(textureDesc
            .setDebugName("Feedback1")
            .setFormat(nvrhi::Format::RGBA16_FLOAT)
            .setIsRenderTarget(false)
            .setIsUAV(true)
            .setUseClearValue(false)
            .setInitialState(nvrhi::ResourceStates::UnorderedAccess));

        m_renderTargets.feedback2 = GetDevice()->createTexture(textureDesc
            .setDebugName("Feedback2"));

        m_renderTargets.motionVectors = GetDevice()->createTexture(textureDesc
            .setDebugName("MotionVectors")
            .setFormat(nvrhi::Format::RG16_FLOAT)
            .setIsRenderTarget(true)
            .setIsUAV(false)
            .setUseClearValue(false)
            .setInitialState(nvrhi::ResourceStates::RenderTarget));

        m_renderTargets.depthFramebufferFactory = std::make_shared<engine::FramebufferFactory>(GetDevice());
        m_renderTargets.depthFramebufferFactory->DepthTarget = m_renderTargets.depth;

        m_renderTargets.framebufferFactory = std::make_shared<engine::FramebufferFactory>(GetDevice());
        m_renderTargets.framebufferFactory->RenderTargets.push_back(m_renderTargets.color);
        m_renderTargets.framebufferFactory->DepthTarget = m_renderTargets.depth;
    }
        
    void CreateRenderPasses()
    {
        m_skyPass = std::make_shared<donut::render::SkyPass>(GetDevice(), m_shaderFactory,
            m_commonPasses, m_renderTargets.framebufferFactory, m_view);

        donut::render::TemporalAntiAliasingPass::CreateParameters taaParams;
        taaParams.sourceDepth = m_renderTargets.depth;
        taaParams.motionVectors = m_renderTargets.motionVectors;
        taaParams.unresolvedColor = m_renderTargets.color;
        taaParams.resolvedColor = m_renderTargets.resolvedColor;
        taaParams.feedback1 = m_renderTargets.feedback1;
        taaParams.feedback2 = m_renderTargets.feedback2;
        m_taaPass = std::make_unique<donut::render::TemporalAntiAliasingPass>(GetDevice(),
            m_shaderFactory, m_commonPasses, m_view, taaParams);
    }

    bool KeyboardUpdate(int key, int scancode, int action, int mods) override
    {
        m_camera.KeyboardUpdate(key, scancode, action, mods);

        return ImGui_Renderer::KeyboardUpdate(key, scancode, action, mods);
    }

    bool MousePosUpdate(double xpos, double ypos) override
    {
        if (ImGui_Renderer::MousePosUpdate(xpos, ypos))
            return true;

        m_camera.MousePosUpdate(xpos, ypos);
            
        return true;
    }

    bool MouseButtonUpdate(int button, int action, int mods) override
    {
        if (ImGui_Renderer::MouseButtonUpdate(button, action, mods))
            return true;

        m_camera.MouseButtonUpdate(button, action, mods);

        return true;
    }

    bool MouseScrollUpdate(double xoffset, double yoffset) override
    {
        if (ImGui_Renderer::MouseScrollUpdate(xoffset, yoffset))
            return true;

        m_camera.MouseScrollUpdate(xoffset, yoffset);

        return true;
    }

    void Animate(float fElapsedTimeSeconds) override
    {
        ImGui_Renderer::Animate(fElapsedTimeSeconds);
        m_camera.Animate(fElapsedTimeSeconds);
    }

    void BackBufferResizing() override
    { 
        ImGui_Renderer::BackBufferResizing();
        m_bindingCache->Clear();
        m_renderTargets = RenderTargets();
    }

    void SetupView(nvrhi::FramebufferInfoEx const& fbinfo)
    {
        m_previousView = m_view;

        dm::affine3 viewMatrix = m_camera.GetWorldToViewMatrix();
        float const aspectRatio = float(fbinfo.width) / float(fbinfo.height);
        float verticalFov = dm::radians(60.f);
        float zNear = 0.01f;
        m_camera.GetSceneCameraProjectionParams(verticalFov, zNear);

        dm::float4x4 const projMatrix = dm::perspProjD3DStyleReverse(verticalFov, aspectRatio, zNear);

        m_view.SetMatrices(viewMatrix, projMatrix);
        m_view.SetViewport(nvrhi::Viewport(fbinfo.width, fbinfo.height));
        m_view.UpdateCache();

        if (m_camera.IsThirdPersonActive())
            m_camera.GetThirdPersonCamera().SetView(m_view);

        if (GetDeviceManager()->GetFrameIndex() == 0)
            m_previousView = m_view;
    }

    void RenderScene(nvrhi::ICommandList* commandList)
    {
        render::SkyParameters skyParameters{};
        skyParameters.brightness = 0.5f;
        m_skyPass->Render(commandList, m_view, *m_light, skyParameters);

        render::InstancedOpaqueDrawStrategy opaqueDrawStrategy;
        render::TransparentDrawStrategy transparentDrawStrategy;

            if (m_useDepthPrepass)
            {
                m_prePassTimer.beginQuery(m_commandList);

                render::DepthPass::Context depthContext;
                render::RenderCompositeView(commandList, &m_view, &m_view, *m_renderTargets.depthFramebufferFactory,
                    m_scene->GetSceneGraph()->GetRootNode(), opaqueDrawStrategy, *m_depthPass,
                    depthContext, "Depth Pre-pass");

                m_prePassTimer.endQuery(m_commandList);
            }

        NtcForwardShadingPass::Context forwardContext;
        m_ntcForwardShadingPass->PrepareLights(commandList, { m_light },
            skyParameters.skyColor * skyParameters.brightness,
            skyParameters.groundColor * skyParameters.brightness);
        m_ntcForwardShadingPass->PreparePass(forwardContext, commandList, GetFrameIndex(),
            m_useSTF, m_stfFilterMode, m_useDepthPrepass, !m_inferenceOnSample);

        m_renderPassTimer.beginQuery(m_commandList);

        render::RenderCompositeView(commandList, &m_view, &m_view, *m_renderTargets.framebufferFactory,
            m_scene->GetSceneGraph()->GetRootNode(), opaqueDrawStrategy, *m_ntcForwardShadingPass,
            forwardContext, "Opaque");

        render::RenderCompositeView(commandList, &m_view, &m_view, *m_renderTargets.framebufferFactory,
            m_scene->GetSceneGraph()->GetRootNode(), transparentDrawStrategy, *m_ntcForwardShadingPass,
            forwardContext, "Transparent");

        m_renderPassTimer.endQuery(m_commandList);
    }

    void SaveScreenshot()
    {
        engine::SaveTextureToFile(
            GetDevice(),
            m_commonPasses.get(),
            GetDeviceManager()->GetCurrentBackBuffer(),
            nvrhi::ResourceStates::Unknown,
            m_screenshotFileName.c_str(),
            /* saveAlphaChannel = */ false);

        m_screenshotFileName.clear();
    }
    
    bool ShouldRenderUnfocused() override
    {
        return true;
    }

    void Render(nvrhi::IFramebuffer* framebuffer) override
    {
        nvrhi::FramebufferInfoEx const& fbinfo = framebuffer->getFramebufferInfo();

        SetupView(fbinfo);

        if (!m_renderTargets.color)
        {
            CreateRenderTargets(fbinfo.width, fbinfo.height);
            CreateRenderPasses();
            m_previousFrameValid = false;
        }

        // This sequence depends on CreateRenderPasses above, which in turn depends on SetupView...
        m_taaPass->AdvanceFrame();
        m_view.SetPixelOffset(m_aaMode == AntiAliasingMode::Off
            ? dm::float2::zero()
            : m_taaPass->GetCurrentPixelOffset());
        m_view.UpdateCache();

        // Initialize or resize the DLSS feature
#if NTC_WITH_DLSS
        if (m_aaMode == AntiAliasingMode::DLSS)
        {
            if (m_DLSS)
            {
                m_DLSS->SetRenderSize(fbinfo.width, fbinfo.height, fbinfo.width, fbinfo.height);
                if (!m_DLSS->IsAvailable())
                    m_aaMode = AntiAliasingMode::TAA;
            }
            else
                m_aaMode = AntiAliasingMode::TAA;
        }
#endif

        m_commandList->open();

        m_commandList->clearDepthStencilTexture(m_renderTargets.depth, nvrhi::AllSubresources, true, 0.f, false, 0);
        m_commandList->clearTextureFloat(m_renderTargets.color, nvrhi::AllSubresources, nvrhi::Color(0.f));

        RenderScene(m_commandList);

        switch (m_aaMode)
        {
            case AntiAliasingMode::Off:
                m_commonPasses->BlitTexture(m_commandList, framebuffer, m_renderTargets.color, m_bindingCache.get());
                break;
            case AntiAliasingMode::TAA: {
                m_taaPass->RenderMotionVectors(m_commandList, m_view, m_previousView);
                donut::render::TemporalAntiAliasingParameters taaParams;
                m_taaPass->TemporalResolve(m_commandList, taaParams, m_previousFrameValid, m_view, m_view);
                m_commonPasses->BlitTexture(m_commandList, framebuffer, m_renderTargets.resolvedColor, m_bindingCache.get());
                break;
            }
#if NTC_WITH_DLSS
            case AntiAliasingMode::DLSS:
                m_taaPass->RenderMotionVectors(m_commandList, m_view, m_previousView);
                m_DLSS->Render(m_commandList, m_renderTargets, 1.f, !m_previousFrameValid, m_view, m_previousView);
                m_commonPasses->BlitTexture(m_commandList, framebuffer, m_renderTargets.resolvedColor, m_bindingCache.get());
                break;
#endif
        }
        
        m_commandList->close();
        GetDevice()->executeCommandList(m_commandList);

        m_prePassTimer.update();
        m_renderPassTimer.update();
        m_previousFrameValid = true;

        if (!m_screenshotFileName.empty() && !m_screenshotWithUI)
            SaveScreenshot();

        ImGui_Renderer::Render(framebuffer);

        if (!m_screenshotFileName.empty() && m_screenshotWithUI)
            SaveScreenshot();
    }

    static char const* BoolToUIString(bool value)
    {
        return value ? "YES" : "NO";
    }

    void buildUI() override
    {
        ImGui::PushFont(m_primaryFont->GetScaledFont());
        float const fontSize = ImGui::GetFontSize();

        ImGui::SetNextWindowPos(ImVec2(fontSize * 0.6f, fontSize * 0.6f), 0);
        if (ImGui::Begin("Settings", nullptr, ImGuiWindowFlags_AlwaysAutoResize))
        {
            ImGui::PushFont(m_largerFont->GetScaledFont());

            char const* textureType = "";
            if (g_options.referenceMaterials)
                textureType = "Reference Textures (PNGs etc.)";
            else if (m_inferenceOnSample)
                textureType = "NTC Inference on Sample";
            else if (g_options.blockCompression)
                textureType = "NTC Transcoded to BCn";
            else
                textureType = "NTC Decompressed on Load";

            ImGui::TextUnformatted(textureType);
            
            size_t textureMemorySize = 0;
            if (g_options.referenceMaterials)
            {
                textureMemorySize = m_referenceTextureMemorySize;
            }
            else
            {
                if (m_inferenceOnSample)
                    textureMemorySize = m_ntcTextureMemorySize;
                else
                    textureMemorySize += m_transcodedTextureMemorySize;
            }
            ImGui::Text("Texture Memory: %.2f MB", float(textureMemorySize) / 1048576.f);
            
            auto renderTime = m_renderPassTimer.getAverageTime();
            if (renderTime.has_value())
            {
                ImGui::Text("Forward Pass Time: %.2f ms", renderTime.value() * 1e3f);
            }

            ImGui::PopFont();
            
            if (m_useDepthPrepass)
            {
                auto prePassTime = m_prePassTimer.getAverageTime();
                if (prePassTime.has_value())
                {
                    ImGui::Text("Depth pre-pass time: %.2f ms", prePassTime.value() * 1e3f);
                }
            }
            else
            {
                ImGui::TextUnformatted("Depth pre-pass time: N/A");
            }

            float const frameTime = GetDeviceManager()->GetAverageFrameTimeSeconds();
            float const framesPerSecond = (frameTime > 0.f) ? 1.f / frameTime : 0.f;
            ImGui::Text("Frame Rate: %.1f FPS", framesPerSecond);

            ImGui::Text("GPU: %s", GetDeviceManager()->GetRendererString());

            ImGui::Text("CoopVec Support: Int8 (%s), FP8 (%s)",
                BoolToUIString(m_materialLoader->IsCooperativeVectorInt8Supported()),
                BoolToUIString(m_materialLoader->IsCooperativeVectorFP8Supported()));

            ImGui::Separator();


            ImGui::PushItemWidth(fontSize * 9.5f);
            if (ImGui::BeginCombo("Camera", m_camera.IsSceneCameraActive()
                ? m_camera.GetSceneCamera()->GetName().c_str()
                : m_camera.IsThirdPersonActive() ? "Orbiting" : "First-Person"))
            {
                if (ImGui::Selectable("Orbiting", m_camera.IsThirdPersonActive()))
                {
                    m_camera.SwitchToThirdPerson();
                }
                if (ImGui::Selectable("First-Person", m_camera.IsFirstPersonActive()))
                {
                    m_camera.SwitchToFirstPerson();
                }
                for (auto const& camera : m_scene->GetSceneGraph()->GetCameras())
                {
                    if (ImGui::Selectable(camera->GetName().c_str(), camera == m_camera.GetSceneCamera()))
                        m_camera.SwitchToSceneCamera(camera);
                }
                ImGui::EndCombo();
            }
            ImGui::PopItemWidth();

            if (ImGui::Checkbox("VSync", &m_enableVsync))
                GetDeviceManager()->SetVsyncEnabled(m_enableVsync);

            if (!g_options.referenceMaterials)
            {
                // If one of the modes is unavailable, disable the checkbox and force to use the other mode
                bool const disableSelection = !g_options.inferenceOnLoad || !g_options.inferenceOnSample;
                ImGui::BeginDisabled(disableSelection);
                ImGui::Checkbox("Inference On Sample", &m_inferenceOnSample);
                ImGui::EndDisabled();

                if (disableSelection)
                    m_inferenceOnSample = g_options.inferenceOnSample;
            }

            bool effectiveUseSTF = m_inferenceOnSample ? true : m_useSTF;
            ImGui::BeginDisabled(m_inferenceOnSample);
            ImGui::Checkbox("Use STF", &effectiveUseSTF);
            ImGui::EndDisabled();
            if (!m_inferenceOnSample)
                m_useSTF = effectiveUseSTF;

            {
                ImGui::BeginDisabled(!effectiveUseSTF);
                ImGui::PushItemWidth(fontSize * 6.f);
               
                // The combo assumes a specific set of filter constant values, validate that.
                ImGui::Combo("STF Filter Mode", &m_stfFilterMode, "Point\0Linear\0Cubic\0Gaussian\0");
                static_assert(STF_FILTER_TYPE_POINT == 0);
                static_assert(STF_FILTER_TYPE_LINEAR == 1);
                static_assert(STF_FILTER_TYPE_CUBIC == 2);
                static_assert(STF_FILTER_TYPE_GAUSSIAN == 3);

                ImGui::PopItemWidth();
                ImGui::EndDisabled();
            }

            ImGui::Separator();

            ImGui::Checkbox("Depth Pre-pass", &m_useDepthPrepass);

            ImGui::TextUnformatted("Anti-aliasing:");
            if (ImGui::RadioButton("Off", m_aaMode == AntiAliasingMode::Off))
            {
                m_aaMode = AntiAliasingMode::Off;
                m_previousFrameValid = false;
            }
            ImGui::SameLine();
            if (ImGui::RadioButton("TAA", m_aaMode == AntiAliasingMode::TAA))
            {
                m_aaMode = AntiAliasingMode::TAA;
                m_previousFrameValid = false;
            }
#if NTC_WITH_DLSS
            ImGui::SameLine();
            ImGui::BeginDisabled(!m_DLSS);
            if (ImGui::RadioButton("DLSS", m_aaMode == AntiAliasingMode::DLSS))
            {
                m_aaMode = AntiAliasingMode::DLSS;
                m_previousFrameValid = false;
            }
            ImGui::EndDisabled();
#endif

            if (ImGui::Button("Save Screenshot..."))
            {
                char const* filters = "Image Files (BMP, PNG, JPG, TGA)\0*.bmp;*.png;*.jpg;*.jpeg;*.tga\0All Files\0*.*\0";
                if (!app::FileDialog(false, filters, m_screenshotFileName))
                    m_screenshotFileName.clear();
            }
            ImGui::SameLine();
            ImGui::Checkbox("Include UI", &m_screenshotWithUI);
        }
        ImGui::End();
        ImGui::PopFont();
    }
};

#ifdef WIN32
int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow)
#else
int main(int __argc, const char** __argv)
#endif
{
    donut::log::SetErrorMessageCaption(g_ApplicationName);
    
    if (!ProcessCommandLine(__argc, (const char**)__argv))
        return 1;
        
#if NTC_WITH_DX12 && NTC_WITH_VULKAN
    const nvrhi::GraphicsAPI graphicsApi = g_options.useDX12
        ? nvrhi::GraphicsAPI::D3D12
        : nvrhi::GraphicsAPI::VULKAN;
#elif NTC_WITH_VULKAN
    const nvrhi::GraphicsAPI graphicsApi = nvrhi::GraphicsAPI::VULKAN;
#else
    const nvrhi::GraphicsAPI graphicsApi = nvrhi::GraphicsAPI::D3D12;
#endif

    std::unique_ptr<app::DeviceManager> deviceManager = std::unique_ptr<app::DeviceManager>(
        app::DeviceManager::Create(graphicsApi));

    app::DeviceCreationParameters deviceParams;
    deviceParams.infoLogSeverity = log::Severity::None;
    deviceParams.vsyncEnabled = false;
    deviceParams.backBufferWidth = 1920;
    deviceParams.backBufferHeight = 1080;
    deviceParams.adapterIndex = g_options.adapterIndex;
    deviceParams.enableDebugRuntime = g_options.debug;
    deviceParams.enableNvrhiValidationLayer = g_options.debug;
    deviceParams.enablePerMonitorDPI = true;
    deviceParams.supportExplicitDisplayScaling = true;

    SetNtcGraphicsDeviceParameters(deviceParams, graphicsApi, false, g_ApplicationName);
#if NTC_WITH_DLSS && NTC_WITH_VULKAN
    if (graphicsApi == nvrhi::GraphicsAPI::VULKAN)
    {
        DLSS::GetRequiredVulkanExtensions(deviceParams.optionalVulkanInstanceExtensions, deviceParams.optionalVulkanDeviceExtensions);
    }
#endif

    if (!deviceManager->CreateWindowDeviceAndSwapChain(deviceParams, g_ApplicationName))
    {
        log::fatal("Cannot initialize a graphics device with the requested parameters");
        return 1;
    }
    
    char windowTitle[256];
    snprintf(windowTitle, sizeof windowTitle, "%s (%s, %s)", g_ApplicationName, 
        nvrhi::utils::GraphicsAPIToString(graphicsApi), deviceManager->GetRendererString());
    deviceManager->SetWindowTitle(windowTitle);

    {
        NtcSceneRenderer example(deviceManager.get());
        if (example.Init())
        {
            deviceManager->AddRenderPassToBack(&example);
            deviceManager->RunMessageLoop();
            deviceManager->RemoveRenderPass(&example);
        }
    }

    deviceManager->Shutdown();

    return 0;
}
