# NTC Compression Settings and Image Quality

## Main concepts

NTC compresses multiple textures, with up to 16 channels total, into a single neural representation. The decompression process produces all channels at once. Contents of one channel may affect image quality in other channels, and sometimes image features can leak between channels. There's almost always some compression error, and the only case when it's guaranteed not to happen is when the entire channel has the same constant value.

Compression quality is primarily driven by the bits-per-pixel (BPP) setting. It can be specified on the command line to `ntc-cli` using the `-b <bpp>` or `--bitsPerPixel <bpp>` argument, or entered using a slider in the NTC Explorer. Adding channels with a fixed BPP value will generally lead to lower compression quality because the amount of compressed information remains the same, while the amount of original information increases, therefore, the compression becomes more lossy.

Image quality is evaluated at the end of every compression run and is reported as the Peak Signal-to-Noise Ratio (PSNR), measured in decibels (dB). Higher PSNR values mean higher quality. For most uses, a PSNR of 35-40 dB can be considered sufficient. A PSNR of 50 dB is perceptually lossless.

## Adaptive compression

The relationship between BPP and PSNR values is highly dependent on the texture data. Less detailed textures are easier to encode and will produce higher PSNR values with the same BPP than more detailed textures. Therefore, it can be difficult to guess which BPP setting should be used for each material to reach the desired image quality.

To make finding the optimal BPP values easier, NTC provides the adaptive compression mode. In this mode, the user specifies a target PSNR vaue that they want to reach, and the CLI tool automatically determines the minimum BPP value that will reach that quality level. This mode is enabled using the `-p <psnr>` or `--targetPsnr <psnr>` argument.

Unfortunately, it is not easy (or maybe even impossible) to algorithmically predict which BPP is optimal based on just the texture data without completing the compression process. Even partial compression runs don't provide enough information for that. So, the CLI tool will perform several full compression runs to locate the optimal BPP for each materials, making the compression process approximately 5x slower.

## HDR images

NTC supports compression of High Dynamic Range images, i.e. those which are stored with more than 8 bits per channel per pixel and can encode channel values greater than 1.0. Internally, all color data is represented as FP16 values, but true HDR images do not work well with the neural decoder - and to work around that, they are converted to the Hybrid Log-Gamma (HLG) color space before compression and linearized after decompression. The conversion is done by the library and enabled automatically in the CLI tool for all EXR images.

The color space conversion used for HDR images has two consequences:

1. Channel values with high magnitude are represented with lower accuracy.
2. The PSNR values computed for HDR images and reported by LibNTC are not correctly normalized and can only be compared to such values from other runs, not from other image comparison methods.

## MIP chains

NTC can compress textures with any number of MIP levels, from just the highest LOD to a full MIP chain all the way down to 1x1 pixel. For that, several latent tensors can be used. Texture MIP levels are combined into groups of 2-4 and each group is compressed into one pair of latent tensors (high-res and low-res ones).

Because of this grouping and potentially suboptimal sampling during compression, texture MIP levels may have somewhat different compression quality in one texture set. Mostly, this difference shows up in LODs 3 and above, not affecting the high-detail levels.

## Alpha mask channel

In most cases, NTC makes no distinction between individual texture channels, and doesn't transform their data in any way beyond user-specified color space conversions. But there is one exception: the alpha mask channel.

Alpha mask channel can be specified in the [manifest](Manifest.md) by providing an `AlphaMask` (or equivalent) semantic to a texture. There can only be a single mask channel in the texture set. When it is specified, it will be treated by the compression process in a special way: the 0.0 and 1.0 values are preserved with higher accuracy than in other channels. Additionally, if the `--discardMaskedOutPixels` option is specified on the command line to `ntc-cli`, the non-mask channels of all pixels where the mask is 0.0 will be ignored. This results in higher image quality for the other pixels; at the same time, the ignored channels of decompressed masked out pixels contain garbage. The Explorer app also has check boxes for enabling the special processing of the mask channel and for ignoring the other channels.

Note that while NTC provides special treatment for preserving the alpha mask channel, in many cases it is desirable to store the mask (or opacity) channel separately, either as a transcoded BC4 texture in video memory or on disk also. There are two main reasons for that:

1. Performance. Alpha mask textures are often used in passes that don't need full material evaluation, such as Z pre-pass or ray tracing any-hit shaders. Sampling a BC4 texture is much faster than decoding an NTC texture set and then ignoring most of the outputs. Plus, it is possible to use hardware filtering with BC4 textures, which produces sharp alpha cutout lines without noise (compared to using STF for the same task which dithers the edges).
2. Image quality. In some cases, there are smooth gradients on the opacity texture, and some value like 0.5 is used for alpha cutout. NTC introduces a small amount of noise to textures, and under a strong nonlinearity like the alpha cutoff, that noise can look like a static boiling pattern. This is not a typical case, and regular cutouts like tree leaves work fine when compressed with NTC.
