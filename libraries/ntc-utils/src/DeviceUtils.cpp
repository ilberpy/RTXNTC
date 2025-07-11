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

#if NTC_WITH_DX12
#include <directx/d3d12.h>
extern "C"
{
    _declspec(dllexport) extern const unsigned int D3D12SDKVersion = D3D12_PREVIEW_SDK_VERSION;
    _declspec(dllexport) extern const char* D3D12SDKPath = ".\\d3d12\\";
}
#endif

#include <ntc-utils/DeviceUtils.h>

#include <libntc/ntc.h>
#include <donut/app/DeviceManager.h>
#include <donut/core/log.h>

#if NTC_WITH_DX12
static bool g_dx12DeveloperModeEnabled = false;
#endif
#if NTC_WITH_VULKAN
static bool g_vulkanDP4aSupported = false;
static bool g_vulkanFloat16Supported = false;
#endif

bool IsDP4aSupported(nvrhi::IDevice* device)
{
#if NTC_WITH_VULKAN
    if (device->getGraphicsAPI() == nvrhi::GraphicsAPI::VULKAN)
        return g_vulkanDP4aSupported;
#endif

#if NTC_WITH_DX12
    if (device->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D12)
    {
        D3D12_FEATURE_DATA_SHADER_MODEL featureData{};
        featureData.HighestShaderModel = D3D_SHADER_MODEL_6_7;
        ID3D12Device* d3d12Device = device->getNativeObject(nvrhi::ObjectTypes::D3D12_Device);
        HRESULT hr = d3d12Device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL, &featureData, sizeof(featureData));
        if (SUCCEEDED(hr))
        {
            // dot4add is a required feature of SM6.4
            // https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/hlsl-shader-model-6-4-features-for-direct3d-12
            return featureData.HighestShaderModel >= D3D_SHADER_MODEL_6_4;
        }    
    }
#endif

    return false;
}

bool IsFloat16Supported(nvrhi::IDevice* device)
{
#if NTC_WITH_VULKAN
    if (device->getGraphicsAPI() == nvrhi::GraphicsAPI::VULKAN)
        return g_vulkanFloat16Supported;
#endif

#if NTC_WITH_DX12
    if (device->getGraphicsAPI() == nvrhi::GraphicsAPI::D3D12)
    {
        D3D12_FEATURE_DATA_D3D12_OPTIONS4 featureData{};
        ID3D12Device* d3d12Device = device->getNativeObject(nvrhi::ObjectTypes::D3D12_Device);
        HRESULT hr = d3d12Device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS4, &featureData, sizeof(featureData));
        if (SUCCEEDED(hr))
        {
            return featureData.Native16BitShaderOpsSupported;
        }    
    }
#endif

    return false;
}

bool IsDX12DeveloperModeEnabled()
{
#if NTC_WITH_DX12
    return g_dx12DeveloperModeEnabled;
#else
    return false;
#endif
}

void SetNtcGraphicsDeviceParameters(
    donut::app::DeviceCreationParameters& deviceParams,
    nvrhi::GraphicsAPI graphicsApi,
    bool enableSharedMemory,
    char const* windowTitle)
{
#if NTC_WITH_VULKAN
    if (graphicsApi == nvrhi::GraphicsAPI::VULKAN)
    {
        if (enableSharedMemory)
        {
#ifdef _WIN32
            deviceParams.requiredVulkanDeviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_WIN32_EXTENSION_NAME);
#else
            deviceParams.requiredVulkanDeviceExtensions.push_back(VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME);
#endif
        }
        deviceParams.optionalVulkanDeviceExtensions.push_back(VK_NV_COOPERATIVE_VECTOR_EXTENSION_NAME);
        deviceParams.optionalVulkanDeviceExtensions.push_back(VK_EXT_SHADER_DEMOTE_TO_HELPER_INVOCATION_EXTENSION_NAME);
        deviceParams.optionalVulkanDeviceExtensions.push_back(VK_EXT_SHADER_REPLICATED_COMPOSITES_EXTENSION_NAME);

        // vkCmdCopyImage: Dest image pRegion[0] x-dimension offset [0] + extent [4] exceeds subResource width [2]
        // vkCmdCopyImage: Dest image pRegion[0] y-dimension offset [0] + extent [4] exceeds subResource height [2]
        // These errors happen during copies from block textures to BCn textures at the last 2 mips, no way around it.
        deviceParams.ignoredVulkanValidationMessageLocations.push_back(0x38b5face);
        deviceParams.ignoredVulkanValidationMessageLocations.push_back(0x4bb17a0e);

        // The following warnings are related to the Cooperative Vector extension that the validation layers don't know.
        // SPIR-V module not valid: Invalid capability operand: 5394
        deviceParams.ignoredVulkanValidationMessageLocations.push_back(0xa5625282);
        // vkCreateShaderModule(): A SPIR-V Capability (Unhandled OpCapability) was declared that is not supported by Vulkan.
        deviceParams.ignoredVulkanValidationMessageLocations.push_back(0x2c00a3d6);
        deviceParams.ignoredVulkanValidationMessageLocations.push_back(0x605314fa);
        // A SPIR-V Extension (SPV_NV_cooperative_vector) was declared that is not supported by Vulkan.
        deviceParams.ignoredVulkanValidationMessageLocations.push_back(0x938b32);
        // Device Extension VK_NV_cooperative_vector is not supported by this layer
        deviceParams.ignoredVulkanValidationMessageLocations.push_back(0x79de34d4);
        // vkCreateDevice: pCreateInfo->pNext chain includes a structure with unknown VkStructureType (1000491000)
        // That's VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_FEATURES_NV
        deviceParams.ignoredVulkanValidationMessageLocations.push_back(0x901f59ec);

        // fragment shader writes to output location 1 with no matching attachment
        // This happens in the forward shading pass for transmissive materials. Difficult to work around.
        deviceParams.ignoredVulkanValidationMessageLocations.push_back(0x609a13b);
        
                
        // Add feature structures querying for cooperative vector support and DP4a support
        static VkPhysicalDeviceCooperativeVectorFeaturesNV cooperativeVectorFeatures{};
        cooperativeVectorFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_VECTOR_FEATURES_NV;
        static VkPhysicalDeviceShaderReplicatedCompositesFeaturesEXT replicatedCompositesFeatures{};
        replicatedCompositesFeatures.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_REPLICATED_COMPOSITES_FEATURES_EXT;
        replicatedCompositesFeatures.pNext = &cooperativeVectorFeatures;
        static VkPhysicalDeviceVulkan11Features vulkan11Features{};
        vulkan11Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
        vulkan11Features.pNext = &replicatedCompositesFeatures;
        static VkPhysicalDeviceVulkan12Features vulkan12Features{};
        vulkan12Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
        vulkan12Features.pNext = &vulkan11Features;
        static VkPhysicalDeviceVulkan13Features vulkan13Features{};
        vulkan13Features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
        vulkan13Features.pNext = &vulkan12Features;
        deviceParams.physicalDeviceFeatures2Extensions = &vulkan13Features;
        
        // Set the callback to modify some bits in VkDeviceCreateInfo before creating the device
        deviceParams.deviceCreateInfoCallback = [](VkDeviceCreateInfo& info)
        {
            const_cast<VkPhysicalDeviceFeatures*>(info.pEnabledFeatures)->shaderInt16 = true;
            const_cast<VkPhysicalDeviceFeatures*>(info.pEnabledFeatures)->fragmentStoresAndAtomics = true;

            // Iterate through the structure chain and find the structures to patch
            VkBaseOutStructure* pCurrent = reinterpret_cast<VkBaseOutStructure*>(&info);
            VkBaseOutStructure* pLast = nullptr;
            while (pCurrent)
            {
                if (pCurrent->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES)
                {
                    reinterpret_cast<VkPhysicalDeviceVulkan11Features*>(pCurrent)->storageBuffer16BitAccess = 
                        vulkan11Features.storageBuffer16BitAccess;
                }

                if (pCurrent->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES)
                {
                    g_vulkanFloat16Supported = vulkan12Features.shaderFloat16;
                    reinterpret_cast<VkPhysicalDeviceVulkan12Features*>(pCurrent)->shaderFloat16 = 
                        g_vulkanFloat16Supported;
                    reinterpret_cast<VkPhysicalDeviceVulkan12Features*>(pCurrent)->vulkanMemoryModel = true;
                    reinterpret_cast<VkPhysicalDeviceVulkan12Features*>(pCurrent)->vulkanMemoryModelDeviceScope = true;
                    reinterpret_cast<VkPhysicalDeviceVulkan12Features*>(pCurrent)->storageBuffer8BitAccess = 
                        vulkan12Features.storageBuffer8BitAccess;
                }

                if (pCurrent->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES)
                {
                    g_vulkanDP4aSupported = vulkan13Features.shaderIntegerDotProduct;
                    reinterpret_cast<VkPhysicalDeviceVulkan13Features*>(pCurrent)->shaderIntegerDotProduct =
                        g_vulkanDP4aSupported;
                    reinterpret_cast<VkPhysicalDeviceVulkan13Features*>(pCurrent)->shaderDemoteToHelperInvocation =
                        vulkan13Features.shaderDemoteToHelperInvocation;
                }

                pLast = pCurrent;
                pCurrent = pCurrent->pNext;
            }

            // If cooperative vector is supported, add a feature structure enabling it on the device
            if (pLast && cooperativeVectorFeatures.cooperativeVector)
            {
                pLast->pNext = reinterpret_cast<VkBaseOutStructure*>(&cooperativeVectorFeatures);
                cooperativeVectorFeatures.pNext = nullptr;
                pLast = pLast->pNext;
            }

            // If replicated composites are supported, add a feature structure enabling it on the device
            if (pLast && replicatedCompositesFeatures.shaderReplicatedComposites)
            {
                pLast->pNext = reinterpret_cast<VkBaseOutStructure*>(&replicatedCompositesFeatures);
                replicatedCompositesFeatures.pNext = nullptr;
                pLast = pLast->pNext;
            }
        };
    }
#endif

#if NTC_WITH_DX12
    g_dx12DeveloperModeEnabled = false;
    if (graphicsApi == nvrhi::GraphicsAPI::D3D12)
    {
        UUID Features[] = { D3D12ExperimentalShaderModels, D3D12CooperativeVectorExperiment };
        HRESULT hr = D3D12EnableExperimentalFeatures(_countof(Features), Features, nullptr, nullptr);

        if (FAILED(hr))
        {
            char const* messageText = 
                "Couldn't enable D3D12 experimental shader models. Cooperative Vector features will not be available.\n"
                "Please make sure that Developer Mode is enabled in the Windows system settings.";

            if (windowTitle)
            {
                MessageBoxA(NULL, messageText, windowTitle, MB_ICONWARNING);
            }
            else
            {
                donut::log::warning("%s", messageText);
            }
        }
        else
        {
            g_dx12DeveloperModeEnabled = true;
        }
    }
#endif
}
