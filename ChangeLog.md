# RTX Neural Texture Compression SDK Change Log

## 0.7.0 BETA

### LibNTC

- Switched the DX12 Cooperative Vector implementation to use the Agility SDK Preview instead of the NVIDIA custom extensions.
- Moved the Cooperative Vector weight layout conversions to happen on the GPU.
- Added support for shuffling inference output channels to make channel mappings more flexible.
- Improved code quality around inference weight layouts.

### Rendering Sample

- Implemented a custom GLTF extension `NV_texture_swizzle` to define the NTC storage for materials.
- Improved the Inference on Feedback mode to transcode tiles in batches.
- Improved the Inference on Sample mode by replacing conditional texture channel usage with constant output channels.
- Added a display of the inference math versions that are being used for materials.

## 0.6.1 BETA

- Improved the Inference on Feedback mode to add support for standby tiles.
- Fixed the rendering mode display in the Renderer sample in the reference mode.
- Implemented handling of `VK_SUBOPTIMAL_KHR` in the SDK apps (https://github.com/NVIDIA-RTX/RTXNTC/issues/3)

## 0.6.0 BETA

Added the Inference on Feedback mode in the Renderer sample.

## 0.5.0 BETA

Initial release.