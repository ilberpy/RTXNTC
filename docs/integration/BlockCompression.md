# Compressing or Transcoding Textures into BCn with LibNTC

LibNTC implements fast compression into BC1-7 using compute shaders. Most of these modes provide quality similar to that achieved with [NVTT3](https://developer.nvidia.com/gpu-accelerated-texture-compression) but compression often runs much faster, which makes it suitable for use at game load time.

Current limitations:
- Signed versions of BC4, BC5 and BC6H are not supported.
- BC6H uses a third party open-source implementation ([knarkowicz/GPURealTimeBC6H](https://github.com/knarkowicz/GPURealTimeBC6H) on GitHub) which doesn't consider all possible modes, reducing the output quality.

Transcoding from NTC into BCn does not require any special support: first, decode the NTC texture set into color textures; then, compress each color texture individually into its target BCn format. LibNTC also implements accelerated transcoding for BC7, see below.

## Implementing BCn Encoding

The integration process is somewhat similar to NTC texture set decompression with graphics APIs, in that the library only provides resources necessary to run the shader, but doesn't launch anything on the GPU.

First, call `IContext::MakeBlockCompressionComputePass(...)` to describe a pass.

```c++
ntc::MakeBlockCompressionComputePassParameters params;
params.srcRect.width = width;
params.srcRect.height = height;
params.dstFormat = ntc::BlockCompressionFormat::BC1;
ntc::ComputePassDesc bcPass;
ntc::Status ntcStatus = m_ntcContext->MakeBlockCompressionComputePass(params, &bcPass);

if (ntcStatus != ntc::Status::Ok)
    // Handle the error
```

After obtaining the compute pass description, use that to create a compute pipeline and bind the input and output resources, as described in the comment to `MakeBlockCompressionComputePass`. The implementation of compute pipeline operation depends on the engine; for an example using NVRHI, see [GraphicsBlockCompressionPass.cpp](/libraries/ntc-utils/src/GraphicsBlockCompressionPass.cpp).

The input to the block compression passes is just a texture object. The BCn compression passes pair well with the graphics decompression passes: the output of the decompression pass can be fed directly into the block compression pass. The output of the block compression passes is also a texture object, but it's 1/4 of the input pixel dimensions and it has one pixel per 4x4 block. The pixel format for the output is `R32G32_UINT` for BC1 or BC4, and `R32G32B32A32_UINT` for all other BC formats. Textures using this pixel format can be copied to native BCn resources directly using a "[reinterpret copy](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/nf-d3d12-id3d12graphicscommandlist-copyresource#reinterpret-copy)" mechanism. In some cases, it may be possible to create texture objects with both UINT and BCn formats in the same memory region to avoid a copy, but in-memory layout compatibility is not guaranteed.

The block compression passes also support specifying an input rectangle and an output offset. Together with partial decompression, this feature can be used to implement incremental transcoding from NTC into BCn.

In the code snippet above, note that some parameters like `bcFormat` are hardcoded. In real integrations, their values should come from an `ITextureMetadata` object, which provides information about a specific texture stored in the NTC texture set.

## BC7 Acceleration

BC6 and BC7 are different from all other BC formats because they support multiple encoding modes and partitioning schemes in each block. LibNTC implements an acceleration scheme for BC7 that relies on analyzing the texture during NTC compression and storing some hints for transcoding into BCn later.

The ahead-of-time analysis process consists of the following steps:

1. Run a "slow" compression pass that produces the acceleration data, by calling `IContext::MakeBlockCompressionComputePass(...)` with `writeAccelerationData = true`. This version of the compression pass needs an extra output buffer of at least `ntc::BlockCompressionAccelerationBufferSize` bytes.
2. Read the contents of the acceleration buffer after executing the compression pass and pass them to `ITextureMetadata::SetBlockCompressionAccelerationData(...)`.
3. Subsequent calls to `MakeBlockCompressionComputePass` with the `texture` parameter specified can use the previously defined acceleration data and will be faster or slower, depending on the `quality` parameter. 
4. Search through the `quality` parameter space to determine the optimal setting for the texture, depending on a desired PSNR value or processing time. The quality generally increases with higher `quality` values, so a binary search algorithm can be used.
5. Store the desired `quality` value in the texture metadata using the `ITextureMetadata::SetBlockCompressionQuality(...)` method. Both the acceleration data and the quality value will be stored in the NTC file when the texture set is serialized.

BC7 acceleration is implemented in the NTC command-line tool and is available through the `--optimizeBC` action. The quality loss can be controlled with the `--bcPsnrThreshold <decibels>` parameter.

## Image Comparison

LibNTC provides a utility pass that can compare two images and compute their per-channel and overall MSE/MSLE values. This pass operates similarly to BCn compression and is described by the `IContext::MakeImageDifferenceComputePass(...)` function. See the comments to this function for the list of resources that need to be bound to the pipeline.

The output of the comparison pass will be in a buffer - read the contents of that buffer and pass them to the `ntc::DecodeImageDifferenceResult(...)` function to calculate the MSE values.

An example implementation of the image comparison pass with NVRHI can be found in [GraphicsImageDifferencePass.cpp](/libraries/ntc-utils/src/GraphicsImageDifferencePass.cpp).
