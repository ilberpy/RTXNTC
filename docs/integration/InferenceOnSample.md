# Implementing Inference on Sample with LibNTC

NTC is designed to support decompression of individual texels in the texture set, which means it can be efficiently used to decompress only the texels needed to render a specific view. In this case, the decompression logic is executed directly in the pixel or ray tracing shader where material textures would normally be sampled. This mode is called Inference on Sample.

Compared to regular texture sampling, decompressing texels from NTC is a relatively expensive operation in terms of computation, and it only returns one unfiltered texel with all material channels at a time. This has two important consequences: 

1. Inference on Sample should only be used on high-performance GPUs that support Cooperative Vector extensions. We provide a fallback implementation that uses DP4a for decompression instead of CoopVec, but that is significantly slower and should only be used for functional validation.
2. Simulating regular trilinear or anisotropic texture filtering with NTC would be prohibitively expensive (although functionally possible), so Inference on Sample should be used in combination with Stochastic Texture Filtering (STF) instead, and filtered by a denoiser or DLSS after shading. See [Filtering After Shading with Stochastic Texture Filtering](https://research.nvidia.com/labs/rtr/publication/pharr2024stochtex/) for more information.

Implementing Inference on Sample in a renderer is relatively straightforward and generally consists of three phases: parsing the texture set file, uploading data to the GPU, and running inference in the shader.

### 1. Parsing the texture set file

The first part is exactly the same as with graphics API decompression, or Inference on Load. Open the input file or create a custom stream object, and use it to construct the `ITextureSetMetadata` object. That object can be used later to query information about textures in the set, dimensions, and so on.

```c++
ntc::FileStreamWrapper inputFile(context);
ntcStatus = context->OpenFile(fileName, /* write = */ false, inputFile.ptr());
if (ntcStatus != ntc::Status::Ok)
    // Handle the error.

ntc::TextureSetMetadataWrapper metadata(context);
ntcStatus = context->CreateTextureSetMetadataFromStream(inputFile, metadata.ptr());
if (ntcStatus != ntc::Status::Ok)
    // Handle the error.
```

Once you have the texture set metadata object, use the `ITextureSetMetadata::GetStreamRangeForLatents(...)`, `ITextureSetMetadata::GetInferenceWeights(...)` and `IContext::MakeInferenceData(...)` methods to obtain the data needed to run inference in the shader. Note the `weightType` parameter; it indicates which version of inference function the application will use, which affects the layout of the weights. When the library determines that the graphics device supports CoopVec, the `MakeInferenceData` will be able to provide weights both for CoopVec decompression functions; when there is no such support, a call to `GetInferenceWeights` or `MakeInferenceData` with those weight types will return `Status::Unsupported`.

```c++
// Select the weight type that matches the version of the inference shader that's being used.
auto const weightType = ntc::InferenceWeightType::GenericInt8;

void const* pWeights = nullptr;
size_t weightSize = 0;
ntcStatus = metadata->GetInferenceWeights(weightType, &pWeights, &weightSize)

if (ntcStatus != ntc::Status::Ok)
    // Handle the error.

ntc::StreamRange streamRange;
ntcStatus = metadata->GetStreamRangeForLatents(firstMip, numMips, streamRange);

if (ntcStatus != ntc::Status::Ok)
    // Handle the error.

ntc::InferenceData inferenceData;
ntcStatus = m_ntcContext->MakeInferenceData(metadata, streamRange, weightType, &inferenceData);

if (ntcStatus != ntc::Status::Ok)
    // Handle the error.
```

### 2. Uploading data to the GPU

Inference on Sample needs three pieces of data, obtained through different ways:

1. The constants describing texture set geometry and other parameters. They are represented by the `NtcTextureSetConstants` structure and populated by `IContext::MakeInferenceData(...)`.
2. The network weights. These weights are provided by `ITextureSetMetadata::GetInferenceWeights(...)` and should be uploaded into a ByteAddressBuffer (nonzero offsets are supported).
3. The latents. These comprise the bulk of the NTC texture set data and can be read directly from the NTC file into a ByteAddressBuffer, without going through the library. The simplest option is to read the entire file into a buffer, but that's somewhat redundant. A better option is to query the range of file data necessary to decode a given range of mip levels by calling `ITextureSetMetadata::GetStreamRangeForLatents(...)`, read only that range into a buffer, and pass that range to `MakeInferenceData(...)` so that the offsets in the constant buffer are calculated correctly.

The code for uploading these buffers depends on the engine and its graphics API abstraction layer. For an example using NVRHI, see the `LoadNtcMaterialForRuntime` function in [NtcSceneRenderer.cpp](/samples/renderer/NtcSceneRenderer.cpp).

### 3. Running inference in the shader

The shader functions necessary to perform Inference on Sample are provided with LibNTC, in the [`libntc/shaders/Inference.hlsli`](/libraries/RTXNTC-Library/include/libntc/shaders/Inference.hlsli) and [`libntc/shaders/InferenceCoopVec.hlsli`](/libraries/RTXNTC-Library/include/libntc/shaders/InferenceCoopVec.hlsli) files. Note that the CoopVec header requires a custom version of the [Slang](https://github.com/shader-slang/slang) compiler with custom versions of the [DXC](https://github.com/microsoft/DirectXShaderCompiler) and [glslang](https://github.com/KhronosGroup/glslang) compiler backends at this time because the CoopVec extensions are not yet standardized in Vulkan or DX12. These compiler builds are provided with the NTC SDK, in the [`RTXNTC-Library/external/slang`](/libraries/RTXNTC-Library/external/slang) folder.

Before including the inference header, the application code needs to define the `NETWORK_VERSION` preprocessor macro with the version of the decompression neural network (MLP) that will be evaluated. There are four versions: small, medium, large, and extra large. The corresponding constants are defined in the [`libntc/shaders/InferenceConstants.h`](/libraries/RTXNTC-Library/include/libntc/shaders/InferenceConstants.h) header that can be included on the host side as well. Each texture set must be decompressed with a matching network version that depends on the compression parameters. The network version can be queried with the `ITextureSetMetadata::GetNetworkVersion()` function, and that value should be used to select the appropriate shader permutation. In order to improve performance and/or reduce implementation complexity, all materials used in a scene should be compressed with the same network version. The network version can be specified in the call to `ITextureSet::SetLatentShape(...)` or through the `--networkVersion` argument to `ntc-cli`.

There are a few main function that perform inference, i.e. compute texture colors for a given texel position. They are called `NtcSampleTextureSet`, `NtcSampleTextureSet_CoopVec_Int8` or `NtcSampleTextureSet_CoopVec_FP8`, differing only in the set of instructions that they use and the weights that they require. They have the same signature:

```c++
template<int NETWORK_VERSION>
bool NtcSampleTextureSet[_CoopVec_{Int8|FP8}](
    NtcTextureSetConstants desc,
    ByteAddressBuffer latentsBuffer, uint latentsOffset,
    ByteAddressBuffer weightsBuffer, uint weightsOffset,
    int2 texel,
    int mipLevel,
    bool convertToLinearColorSpace,
    out float outputs[NtcNetworkParams<NETWORK_VERSION>::OUTPUT_CHANNELS]);
```

The `desc` and the `weightsBuffer` buffers are provided earlier by the `MakeInferenceData` and `GetInferenceWeights` functions, and the `latentsBuffer` buffer contains a portion of the NTC texture file, as described above. The `texel` and `mipLevel` parameters point at the specific texel that needs to be decoded. The `convertToLinearColorSpace` parameter tells if the outputs should be converted to linear color space using texture set metadata, or returned in their storage color spaces. The results are placed into the `outputs` array, in the same order that was used when placing textures in the texture set during compression. There is no shader-side API to distribute the output channels into per-texture vectors; that is up to the application. The simplest solution is to use a fixed mapping from texture semantics to the channel indices, like it's done in the Renderer sample app.

Note that the `NtcSampleTextureSet` function takes an integer texel position and not normalized UV coordinates. Applications should calculate the texel position before calling the NTC function, and obtain the texture set dimensions and mip level count using the `NtcGetTextureDimensions` and `NtcGetTextureMipLevels` functions, respectively.

For a complete example running Inference on Sample combined with Stochastic Texture Filtering, see [`renderer/NtcForwardShadingPass.hlsl`](/samples/renderer/NtcForwardShadingPass.hlsl).

## Streaming Virtual Texture Support

NTC texture sets can be loaded in a per-tile fashion in order to operate in a streaming virtual texture system. The general implementation flow is similar to regular Inference on Sample (see above), but the data loading part is different.

### 1. Determine the necessary slice of the texture set

Normally, streaming virtual texture systems operate on fixed-size texture slices, such as 128x128 pixels. When the renderer determines that a texture tile is needed from a certain mip level, that tile is loaded into the physical texture backing store. With NTC, some special considerations should be made: NTC fuses some mip levels together. For example, when `LatentShape.gridSizeScale == 4`, mip levels 0-3 are compressed into the same latent image, then mip levels 4-5 into another one and so on. A single slice of these latent images represents different size slices on different mip levels. So, if a 128x128 slice of mip 2 is loaded, the same latents can also be used to decode 256x256 pixels from mip 1 and 512x512 pixels from mip 0. Therefore, treating these mip levels separately could be wasteful.

In order to help the application determine the right strategy for combining mip levels, NTC provides the function `ITextureSetMetadata::GetFusedMipLevels(...)` that returns the range of mips that are represented by the same latent image as the given mip level.

### 2. Suballocate buffer memory

Instead of a 2D texture used in regular SVT systems, the backing store for NTC virtual textures is a ByteAddressBuffer. That is because the latents cannot be efficiently packed into common pixel formats in some cases. While this presents some advantages, such as the ability to store textures of different formats or compression ratios in one buffer, it also makes storage management somewhat more complicated.

The simplest approach to memory management for NTC SVT is to use fixed size tiles and fixed size buffer chunks. After the tile size is selected, you can query the buffer chunk size using the `IContext::GetConservativeLatentBufferSize(...)` method. It's called conservative because the actual size of the latent buffer will vary slightly depending on the position of the slice on the texture: slices at the borders will need less data.

A more advanced approach may deal with allocations of random sizes, in which case the chunk size for a specific NTC texture set slice can be queried using the `IContext::MakePartialInferenceData(...)` method with no buffer provided.

### 3. Load the latent data for a slice

The NTC library handles the extraction of the necessary portion of latent data from the input file. It is implemented by `IContext::MakePartialInferenceData(...)`. Note that it requires the input file (or memory stream) to be kept open, and that the data will be placed into a region of CPU memory. Uploading the data to the GPU is left up to the application.

Besides the slice of latent data, `MakePartialInferenceData` populates the same constant buffer structure as `MakeInferenceData`. That structure can be directly passed to the `NtcSampleTextureSet(...)` functions on the shader side. The structure will be slightly different for each slice. It could be redundant to store complete copies of `TextureSetConstants` for all slices, so the application may choose to only store the `highResNeuralMips` and `lowResNeuralMips` components of it for each slice, and patch a global `TextureSetConstants` structure with those per-slice versions before sampling. The global `TextureSetConstants` structure can be obtained with `MakeInferenceData`. _TODO: This is not an ideal solution, could be improved later._

The latent data and the constants should be placed into a known place in the backing store buffer(s) on the GPU.

### 4. Running inference in the shader

Inference in the shader should be done similar to how it's done in the regular, non-streaming case. The only difference is that instead of referencing a single chunk of data representing the entire texture set, the application must use some kind of page table to locate the latents and constants in the backing store buffers before calling `NtcSampleTextureSet(...)`.
