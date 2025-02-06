# BCTest

BCTest is a simple command line application for evaluating the performance and quality of the BCn encoders included with LibNTC, and for comparing them to the encoders provided in the NVIDIA Texture Tools (NVTT3) library.

## Building BCTest

BCTest is configured with the rest of the NTC SDK when the `NTC_WITH_TESTS` CMake variable is `ON` (it is by default).

Support for NVTT3 comparison is not enabled by default. To enable it, do the following:

1. Download and install (Windows) or extract (Linux) the NVTT3 package from the [NVIDIA Developer website](https://developer.nvidia.com/gpu-accelerated-texture-compression). The NTC project is configured for the `30205` version of NVTT3, but it should be easy to modify for newer versions.
2. Set CMake variable `NTC_WITH_NVTT3=ON`.
3. On Linux, provide the path to the extracted package through `NVTT3_SEARCH_PATH`.
4. Configure and build.

## Running BCTest

A typical command line for BCTest looks like this:

```sh
bctest --source <path> --format bc7 --vk --csv <path-to-output-file.csv>
```

This command will find all images in the specified path and compress them into the specified format (`bc1-bc7`). For each image, the MSE or MSLE (for BC6H) and PSNR values are computed and printed out into stdout and the specified CSV file. GPU encoding performance is also measured, but those numbers are not reliable unless GPU clocks are frozen during the test because the test is largely CPU limited due to input image decoding.

If the NVTT3 integration is enabled, the same command will compress the images using NVTT3 and provide its quality metrics, too. You can disable that by adding a `--no-nvtt3` argument to speed up the testing process.

Regression testing for the BCn encoders provided with LibNTC can be done using the CSV files and the `--loadBaseline <file.csv>` argument. That will load the original test results from the specified file and use them to compute differences, saving those into the output CSV file.

Compressed images can be saved as DDS files if the `--output <path>` argument is specified. The original file paths relative to the input are preserved, and each file name gets a suffix: either `.NTC.dds` or `.NVTT.dds`. This is useful for debugging and detailed comparison.

For the full set of command line options, please run `bctest --help`.