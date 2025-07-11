# Decompressing Texture Sets with Graphis APIs (Inference on Load)

LibNTC provides the means to decompress neural texture sets using either DX12 or Vulkan. The library itself doesn't submit any GPU commands or create any GPU resources; instead, it provides description of how to execute the decompression pass on the application side.

To use graphics API functions, first you need to make sure that the context is created with the right graphics API and device object provided, see the [Context](Context.md) section of this guide. The graphics device is currently only used for host-only functions by the library.

To decompress a neural texture set, first create an `ITextureSetMetadata` object from it:

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

The metadata object can be used to inspect the texture set, see what textures are stored in it and in which channels, and to describe the decompression pass. See the example in the [Decompression with CUDA](Compression.md#decompression-with-cuda) section to find out how to enumerate textures in the texture set - those functions are actually provided by the metadata interface. Based on the enumerated textures, you should create GPU texture objects that the decompressed data will be written into.

To obtain the description for the decompression pass:

```c++
ntc::MakeDecompressionComputePassParameters params;
params.textureSetMetadata = metadata;
params.weightType = textureSetMetadata->GetBestSupportedWeightType();

ntc::ComputePassDesc computePass{};
ntc::Status ntcStatus = context->MakeDecompressionComputePass(params, &computePass);    

if (ntcStatus != ntc::Status::Ok)
    // Handle the error.
```

The `ComputePassDesc` returned by `MakeDecompressionComputePass` contains the following information:
- Compute shader bytecode. Note that the bytecode may be different depending on the parameters.
- Constant buffer data.
- Compute dispatch dimensions.

It is the application's responsibility to create the compute pipeline, constant buffer, weight buffer (see below), and fill the buffers with the provided data. The application must also populate the descriptor table (DX12) or descriptor sets (Vk) to bind these resources to the right slots of the compute shader. The bindings are static and described in the comments to `IContext::MakeDecompressionComputePass` in [ntc.h](/libraries/RTXNTC-Library/include/libntc/ntc.h). The actual code needed to perform these operations depends on the graphics API and the abstraction layer (RHI) used by the application. The SDK apps are using [NVRHI](https://github.com/NVIDIAGameWorks/nvrhi), and the implementation of the decompression pass with NVRHI can be found in [GraphicsDecompressionPass.cpp](/libraries/ntc-utils/src/GraphicsDecompressionPass.cpp). 

The most important resources for the decompression pass are provided implicitly: it's the latents buffer which contains most of the NTC texture set file, and the weight buffer that contains the inference weights. The latents data need to be uploaded into a GPU buffer and bound to the decompression pipeline as a `ByteAddressBuffer` (DX12) or `StorageBuffer` (Vk). The exact means of reading the file, uploading it to the GPU, and managing live files in GPU memory is left up to the application; notably, apps can use DirectStorage to load the data.

It's possible to load only a subset of the NTC texture set file for decompression if not all mip levels need to be decompressed. The example above uses the `ntc::EntireStream` constant to specify that the entire file is loaded; to use a subset of the file, first obtain the required stream range for the desired set of mip levels:

```c++
ntc::StreamRange streamRange;
ntcStatus = metadata->GetStreamRangeForLatents(firstMip, numMips, streamRange);

if (ntcStatus != ntc::Status::Ok)
    // Handle the error.
```

Then use the `streamRange` fields `offset` and `size` to read that portion of the input file or stream into a buffer, and provide the range via `MakeDecompressionComputePassParameters::latentStreamRange` to make it provide correct data offsets to the decompression shader.

The data for the weight buffer should be queried from the texture set and depends on the inference math version being used. For automatic selection of the weight version, use the `ITextureSetMetadata::GetBestSupportedWeightType(...)` function as shown above. Note that it might return `Unknown` in a rare but possible case when the texture set does not provide any Int8 data but the GPU does not support CoopVec-FP8 inference.

To query the weights, use the `ITextureSetMetadata::GetInferenceWeights(...)` function as shown below:

```c++
void const* pWeightData = nullptr;
size_t uploadSize = 0;
size_t convertedSize = 0;
ntcStatus = metadata->GetInferenceWeights(params.weightType, &pWeightData, &uploadSize, &convertedSize);

if (ntcStatus != ntc::Status::Ok)
    // Handle the error.

// ... Upload (pWeightData, uploadSize) to 'stagingBuffer' - API/engine dependent ...

if (convertedSize != 0)
{
    // Conversion for CoopVec is needed - perform the conversion on the GPU, moving data from 'stagingBuffer' to 'weightBuffer'.
    // Note that ConvertInferenceWeights takes graphics-API-specific objects for the command list and buffers.
    metadata->ConvertInferenceWeights(params.weightType, commandList, stagingBuffer, stagingOffset, weightBuffer, 0);
}
else
{
    // No conversion is needed - just copy the data from 'stagingBuffer' to 'weightBuffer' (DX12 version below)
    commandList->CopyBufferRegion(weightBuffer, 0, stagingBuffer, stagingOffset, uploadSize);
}
```

Note that the decompression pass supports specifying a rectangle to limit the set of pixels to decompress. This feature can be used to incrementally decompress a large texture in time-constrained scenarios, such as texture streaming while the game is rendering. In order to advance to the next rectangle, you need to call `MakeDecompressionComputePass` again, but that call itself is cheap and doesn't require any new resources to be created.
