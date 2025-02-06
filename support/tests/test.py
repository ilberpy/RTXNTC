#!/usr/bin/python

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import unittest
import os
import sys
import shutil
import numpy
from PIL import Image

try:
    import OpenEXR
    import Imath
    _OPEN_EXR_SUPPORTED = True
except ImportError:
    _OPEN_EXR_SUPPORTED = False

_DP4A_SUPPORTED = False
_FP16_SUPPORTED = False
_COOPVEC_INT8_SUPPORTED = False
_COOPVEC_FP8_SUPPORTED = False

FL_LEGACY = 0
FL_DP4A = 1
FL_FP16 = 2
FL_COOPVEC_INT8 = 3
FL_COOPVEC_FP8 = 4

FL_STRINGS = {
    FL_LEGACY: 'Legacy',
    FL_DP4A: 'DP4a',
    FL_FP16: 'FP16',
    FL_COOPVEC_INT8: 'CoopVecInt8',
    FL_COOPVEC_FP8: 'CoopVecFP8'
}

# add ../../libraries to the path to import ntc
sdkroot = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(sdkroot, 'libraries'))

import ntc

sourceDir = os.path.join(sdkroot, 'assets/materials')
testFilesDir = os.path.join(sdkroot, 'assets/testfiles')
scratchDir = os.path.join(sdkroot, 'assets/testscratch')

def _loadPillowImage(filename):
    image = Image.open(filename)
    return numpy.array(image, dtype=numpy.int16) # Convert to signed int so that (a-b) operation works both ways

def _loadExrImage(filename):
    assert _OPEN_EXR_SUPPORTED
    image = OpenEXR.InputFile(filename)

    header = image.header()
    dataWindow = header['dataWindow']
    width = dataWindow.max.x - dataWindow.min.x + 1
    height = dataWindow.max.y - dataWindow.min.y + 1
    channels = header['channels']
    numChannels = len(channels)
    
    data = numpy.zeros(shape=(width, height, numChannels), dtype=numpy.float32)

    for ch in range(numChannels):
        channelBytes = image.channel('RGBA'[ch], pixel_type=Imath.PixelType(Imath.PixelType.FLOAT))
        channelArray = numpy.frombuffer(channelBytes, dtype=numpy.float32)
        data[:, :, ch] = channelArray.reshape((width, height))

    return data

def _loadAutoImage(filename: str):
    if filename.lower().endswith('.exr'):
        return _loadExrImage(filename)
    else:
        return _loadPillowImage(filename)


class TestCase(unittest.TestCase):

    def setUp(self):
        self.tool = ntc.get_default_tool_path()
        self.prepareScratch()

    def prepareScratch(self):
        if os.path.exists(scratchDir):
            shutil.rmtree(scratchDir)
        os.makedirs(scratchDir)

    def computePSNR(self, img_path1, img_path2, ignoreExtraChannels, hdr=False):
        "Loads two common format image files and returns PSNR between them in dB"
        
        # Load the images
        arr1 = _loadAutoImage(img_path1)
        arr2 = _loadAutoImage(img_path2)

        # Ensure both images have the same dimensions
        self.assertEqual(arr1.shape[0:2], arr2.shape[0:2])

        if len(arr1.shape) == 2: arr1 = numpy.expand_dims(arr1, 2)
        if len(arr2.shape) == 2: arr2 = numpy.expand_dims(arr2, 2)

        # Ensure that image2 has the same or greter number of channels
        if ignoreExtraChannels:
            self.assertGreaterEqual(arr2.shape[2], arr1.shape[2])
            # Delete the extra channels from arr2, if any
            arr2 = numpy.delete(arr2, range(arr1.shape[2], arr2.shape[2]), 2)
        else:
            self.assertEqual(arr2.shape[2], arr1.shape[2])

        # Compute mean squared error (MSE) between the two images
        mse = numpy.mean((arr1 - arr2) ** 2)

        # Compute PSNR from MSE, assuming that image data is in 0-255 integers
        psnr = 10 * numpy.log10(255**2 / mse)

        return psnr

    def assertFileExists(self, path):
        if not os.path.exists(path):
            raise self.failureException(f"'{path}' does not exist")
        
    def assertBetween(self, value, low, high):
        if value < low or value > high:
            raise self.failureException(f'{value} is not between {low} and {high}')


    def compareOutputImages(self, sourceDir, decompressedDir, expectedPsnrValues, toleranceDb, ignoreExtraChannels = False):

        imageNames = [
            'AmbientOcclusion',
            'Color',
            'Displacement',
            'NormalDX',
            'Roughness'
        ]

        for name, expectedPsnr in zip(imageNames, expectedPsnrValues):
            originalImageFileName = os.path.join(sourceDir, f'{name}.jpg')
            decompressedImageFileName = os.path.join(decompressedDir, f'{name}.tga')

            self.assertFileExists(decompressedImageFileName)

            psnr = self.computePSNR(originalImageFileName, decompressedImageFileName, ignoreExtraChannels)

            if expectedPsnr > 0:
                self.assertBetween(psnr, expectedPsnr - toleranceDb, expectedPsnr + toleranceDb)
            else:
                print(f'{name}: actual PSNR = {psnr:.2f} dB')

class DescribeTestCase(TestCase):

    def __str__(self):
        return 'Describe'
        
    def runTest(self):
        ntcFileName = os.path.join(testFilesDir, f'PavingStones070_4bpp_small.ntc')

        args = ntc.Arguments(
            tool=self.tool,
            loadCompressed=ntcFileName,
            describe=True,
            graphicsApi='vk'
        )

        result = ntc.run(args)

        self.assertEqual(result.dimensions, (2048, 2048))
        self.assertEqual(result.channels, 10)
        self.assertEqual(result.mipLevels, 1)
        self.assertEqual(result.networkVersion, 'NTC_NETWORK_SMALL')

        global _DP4A_SUPPORTED, _FP16_SUPPORTED, _COOPVEC_INT8_SUPPORTED, _COOPVEC_FP8_SUPPORTED
        _DP4A_SUPPORTED = 'DP4a' in result.gpuFeatures
        _FP16_SUPPORTED = 'FP16' in result.gpuFeatures
        _COOPVEC_INT8_SUPPORTED = 'CoopVecInt8' in result.gpuFeatures
        _COOPVEC_FP8_SUPPORTED = 'CoopVecFP8' in result.gpuFeatures


class CompressionTestCase(TestCase):

    def __str__(self):
        return 'Compression'
    
    def runTest(self):
        sourceMaterialDir = os.path.join(sourceDir, 'PavingStones070')
        ntcFileName = os.path.join(scratchDir, 'PavingStones070.ntc')
        
        args = ntc.Arguments(
            tool=self.tool,
            loadImages=sourceMaterialDir,
            compress=True,
            decompress=True,
            bitsPerPixel=4.0,
            stepsPerIteration=1000,
            trainingSteps=10000,
            saveCompressed=ntcFileName
        )

        result = ntc.run(args)

        self.assertTrue(os.path.exists(ntcFileName))
        self.assertEqual(result.bitsPerPixel, args.bitsPerPixel)
        self.assertBetween(result.overallPsnr, 28, 32)
        self.assertIsNotNone(result.compressionRuns)
        self.assertEqual(len(result.compressionRuns), 1)
        self.assertEqual(len(result.compressionRuns[0].learningCurve), args.trainingSteps / args.stepsPerIteration)

        decompressedDir = os.path.join(scratchDir, 'output')

        args = ntc.Arguments(
            tool=self.tool,
            loadCompressed=ntcFileName,
            decompress=True,
            saveImages=decompressedDir,
            imageFormat='tga',
            bcFormat='none',
        )

        result = ntc.run(args)

        self.compareOutputImages(sourceMaterialDir, decompressedDir, (33, 29, 39, 28, 35), toleranceDb=2.0)


class HdrCompressionTestCase(TestCase):

    def __str__(self):
        return 'HDR Compression'
    
    @unittest.skipIf(not _OPEN_EXR_SUPPORTED, 'Requires OpenEXR')
    def runTest(self):
        sourceMaterialDir = os.path.join(sourceDir, 'HdrChapel')
        ntcFileName = os.path.join(scratchDir, 'HdrChapel.ntc')
        
        args = ntc.Arguments(
            tool=self.tool,
            loadImages=sourceMaterialDir,
            compress=True,
            decompress=True,
            bitsPerPixel=4.0,
            stepsPerIteration=1000,
            trainingSteps=10000,
            saveCompressed=ntcFileName,
            randomSeed=1337
        )

        result = ntc.run(args)

        self.assertTrue(os.path.exists(ntcFileName))
        self.assertEqual(result.bitsPerPixel, args.bitsPerPixel)
        self.assertBetween(result.overallPsnr, 43, 46)
        self.assertIsNotNone(result.compressionRuns)
        self.assertEqual(len(result.compressionRuns), 1)
        self.assertEqual(len(result.compressionRuns[0].learningCurve), args.trainingSteps / args.stepsPerIteration)

        compressionPsnr = result.overallPsnr

        decompressedDir = os.path.join(scratchDir, 'output')

        args = ntc.Arguments(
            tool=self.tool,
            loadCompressed=ntcFileName,
            decompress=True,
            saveImages=decompressedDir,
            imageFormat='exr',
            bcFormat='none',
        )

        result = ntc.run(args)

        psnr = self.computePSNR(
            os.path.join(sourceMaterialDir, 'Color.exr'),
            os.path.join(decompressedDir, 'Color.exr'),
            ignoreExtraChannels=False)
        
        # Note: this PSNR range is different from the range we expect at compression time.
        # The reason is that we compute PSNR based on raw HDR data here. NTC, on the other hand, computes it
        # in the storage color space, which is HLG, and that gives a different answer. Doing HLG is numpy is very slow.
        # That is technically an NTC bug.
        self.assertBetween(psnr, 40, 44)


class DecompressionTestCase(TestCase):

    def __init__(self, api: str, networkVersion: str, featureLevel: int) -> None:
        super().__init__()
        self.api = api
        self.networkVersion = networkVersion
        self.featureLevel = featureLevel

    def __str__(self):
        return f'Decompression ({self.api}, {self.networkVersion} network, {FL_STRINGS[self.featureLevel]})'

    def runTest(self):
        if self.api == 'dx12' and os.name != 'nt':
            self.skipTest('DX12 is only available on Windows')
        if self.featureLevel >= FL_DP4A and not _DP4A_SUPPORTED:
            self.skipTest('DP4a is not supported')
        if self.featureLevel >= FL_FP16 and not _FP16_SUPPORTED:
            self.skipTest('FP16 is not supported')
        if self.featureLevel >= FL_COOPVEC_INT8 and not _COOPVEC_INT8_SUPPORTED:
            self.skipTest('CoopVec-Int8 is not supported')
        if self.featureLevel >= FL_COOPVEC_FP8 and not _COOPVEC_FP8_SUPPORTED:
            self.skipTest('CoopVec-FP8 is not supported')
        
        sourceMaterialDir = os.path.join(sourceDir, 'PavingStones070')
        ntcFileName = os.path.join(testFilesDir, f'PavingStones070_4bpp_{self.networkVersion}.ntc')
        decompressedDir = os.path.join(scratchDir, 'output')
    
        isCuda = self.api == 'cuda'

        args = ntc.Arguments(
            tool=self.tool,
            loadCompressed=ntcFileName,
            decompress=True,
            saveImages=decompressedDir,
            imageFormat='tga',
            bcFormat='none',
            graphicsApi='' if isCuda else self.api
        )

        if not isCuda:
            args.noDP4a = self.featureLevel < FL_DP4A
            args.noFloat16 = self.featureLevel < FL_FP16
            args.noCoopVecInt8 = self.featureLevel != FL_COOPVEC_INT8
            args.noCoopVecFP8 = self.featureLevel != FL_COOPVEC_FP8

        result = ntc.run(args)
        
        if isCuda:
            self.assertEqual(result.graphicsApi, '')
        elif self.api == 'vk':
            self.assertEqual(result.graphicsApi, 'Vulkan')
        elif self.api == 'dx12':
            self.assertEqual(result.graphicsApi, 'D3D12')
        
        if self.networkVersion in ('small', 'medium'):
            expectedPsnr = (33.8, 29.8, 40.3, 29.6, 36.1)
        else:
            expectedPsnr = (34.3, 30.4, 40.4, 29.5, 35.8)
            
        self.compareOutputImages(sourceMaterialDir, decompressedDir, expectedPsnr, toleranceDb=1.5, ignoreExtraChannels=not isCuda)
        

if __name__ == '__main__':
    suite = unittest.TestSuite()
    # Describe should go first because it also queries the GPU capabilities
    suite.addTest(DescribeTestCase())
    suite.addTest(CompressionTestCase())
    suite.addTest(HdrCompressionTestCase())

    for api in ('cuda', 'vk', 'dx12'):
        for networkVersion in ('small', 'medium', 'large', 'xlarge'):
            if api == 'cuda':
                suite.addTest(DecompressionTestCase(api=api, networkVersion=networkVersion, featureLevel=FL_LEGACY))
            else:
                for featureLevel in (FL_LEGACY, FL_DP4A, FL_FP16, FL_COOPVEC_INT8, FL_COOPVEC_FP8):
                    suite.addTest(DecompressionTestCase(api=api, networkVersion=networkVersion, featureLevel=featureLevel))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
