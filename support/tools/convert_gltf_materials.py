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

import argparse
import copy
import json
import os
import PIL
import PIL.Image
import sys
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

# add ../../libraries to the path to import ntc
sdkroot = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.append(os.path.join(sdkroot, 'libraries'))
import ntc

defaultTool = ntc.get_default_tool_path()

parser = argparse.ArgumentParser()
parser.add_argument('inputFiles', nargs = '+', help = 'Input GLTF files')
group = parser.add_mutually_exclusive_group()
group.add_argument('--bitsPerPixel', default = None, type = float, help = 'Compression bits per pixel value')
group.add_argument('--targetPsnr', default = None, type = float, help = 'Target PSNR value in dB for adaptive compressoin')
parser.add_argument('--maxBitsPerPixel', default = None, type = float, help = 'Maximum bits per pixel value for adaptive compression')
parser.add_argument('--devices', nargs = '*', default = [0], type = int, help = 'List of CUDA devices to use, such as --devices 0 1')
parser.add_argument('--keepManifests', action = 'store_true', help = 'Don\'t delete the manifest files after compression')
parser.add_argument('--output', required = True, help = 'Path to the output directory for the material NTC files')
parser.add_argument('--skipExisting', action = 'store_true', help = 'Skip materials where the NTC file already exists')
parser.add_argument('--tool', default = defaultTool, help = f'Path to the ntc-cli executable, defaults to {defaultTool}')
parser.add_argument('--verbose', action = 'store_true', help = 'Print the commands before executing them')
parser.add_argument('--networkVersion', default='auto', help = 'Network version to use for all materials, small...xlarge or auto')
parser.add_argument('--dryRun', action = 'store_true', help = 'Process materials and write manifest files, but do not compress')
args = parser.parse_args()

if not os.path.isfile(args.tool):
    print(f"The specified tool file '{args.tool}' does not exist.", file = sys.stderr)
    sys.exit(1)

if not os.path.isdir(args.output):
    os.makedirs(args.output)

@dataclass
class TextureParams:
    name: str
    gltfPath: str
    firstChannel: int
    semantics: dict
    bcFormat: str
    sRGB: bool = False
    channelSwizzle: Optional[str] = None

textureTypes = [
    TextureParams(
        name = 'BaseColor', 
        gltfPath = 'pbrMetallicRoughness/baseColorTexture', 
        firstChannel = 0, 
        semantics = { 'Albedo': 'RGB', 'AlphaMask': 'A' },
        bcFormat = 'BC7',
        sRGB = True),
    TextureParams(
        name = 'DiffuseColor', 
        gltfPath = 'extensions/KHR_materials_pbrSpecularGlossiness/diffuseTexture', 
        firstChannel = 0,
        semantics = { 'Albedo': 'RGB', 'AlphaMask': 'A' },
        bcFormat = 'BC7',
        sRGB = True),
    TextureParams(
        name = 'MetallicRoughness',
        gltfPath = 'pbrMetallicRoughness/metallicRoughnessTexture',
        firstChannel = 4, 
        semantics = { 'Metallic': 'R', 'Roughness': 'G' },
        bcFormat = 'BC5',
        channelSwizzle = 'BG'),
    TextureParams(
        name = 'SpecularGlossiness',
        gltfPath = 'extensions/KHR_materials_pbrSpecularGlossiness/specularGlossinessTexture',
        firstChannel = 4, 
        semantics = { 'SpecularColor': 'RGB', 'Glossiness': 'A' },
        bcFormat = 'BC7',
        sRGB = True),
    TextureParams(
        name = 'Normal',
        gltfPath = 'normalTexture',
        firstChannel = 8, 
        semantics = { 'Normal': 'RGB' },
        bcFormat = 'BC7'),
    TextureParams(
        name = 'Occlusion',
        gltfPath = 'occlusionTexture',
        firstChannel = 11, 
        semantics = { 'Occlusion': 'R' },
        bcFormat = 'BC4',
        channelSwizzle = 'R'),
    TextureParams(
        name = 'Emissive',
        gltfPath = 'emissiveTexture',
        firstChannel = 12, 
        semantics = { 'Emissive': 'RGB' },
        bcFormat = 'BC7',
        sRGB = True),
    TextureParams(
        name = 'Transmission',
        gltfPath = 'extensions/KHR_materials_transmission/transmissionTexture',
        firstChannel = 15, 
        semantics = { 'Transmission': 'R' },
        bcFormat = 'BC4',
        channelSwizzle = 'R'),
]

def get_file_basename(fileName):
    head, tail = os.path.split(fileName)
    return tail if tail else head

def get_file_stem(fileName):
    return os.path.splitext(get_file_basename(fileName))[0]

def add_texture_to_manifest(material, manifestTextures, texture: TextureParams, texturesNode, imagesNode, gltfDir, manifestDir):
    node = material
    for part in texture.gltfPath.split('/'):
        node = node.get(part, None)
        if node is None:
            return
    textureIndex = node['index']
    imageIndex = texturesNode[textureIndex]['source']
    image = imagesNode[imageIndex]

    try:
        imageUri = image['uri']
    except KeyError:
        imageName = image['name']
        print(f'  WARNING: Image "{imageName}" doesn\'t have a URI attribute, skipping.')
        print( '    Note: Only "glTF Separate" format models are supported.')
        return

    imageUri = os.path.join(gltfDir, imageUri)
    if (imageUri.endswith('.dds')):
        foundReplacement = False
        for ext in ('.png', 'jpg', '.jpeg', '.tga', '.bmp'):
            newUri = os.path.splitext(imageUri)[0] + ext
            if os.path.exists(newUri):
                imageUri = newUri
                foundReplacement = True
                break
        if not foundReplacement:
            print(f'  WARNING: Couldn\'t find a non-DDS replacement for {imageUri}, skipping.')
            return
    elif not os.path.exists(imageUri):
        print(f'  WARNING: {imageUri} does not exist, skipping.')
        return
    
    print(f'  {texture.name}: {get_file_basename(imageUri)}')
    
    imagePathRelativeToManifest = os.path.relpath(imageUri, manifestDir)
    
    manifestEntry = {
        'fileName': imagePathRelativeToManifest,
        'name': texture.name,
        'isSRGB': texture.sRGB,
        'bcFormat': texture.bcFormat
    }

    semantics = copy.deepcopy(texture.semantics)

    # Find the name of the semantic that uses the Alpha channel, if any
    alphaSemantic = None
    for name, channels in semantics.items():
        if channels == 'A':
            alphaSemantic = name

    # If there is such semantic, see if the image has an alpha channel.
    # Delete the semantic if there is no alpha channel.
    if alphaSemantic is not None:
        with PIL.Image.open(imageUri) as img:
            if img.mode != 'RGBA':
                del semantics[alphaSemantic]
                
    manifestEntry['semantics'] = semantics

    if texture.channelSwizzle is not None:
        manifestEntry['channelSwizzle'] = texture.channelSwizzle

    manifestEntry['firstChannel'] = texture.firstChannel

    manifestTextures.append(manifestEntry)

@dataclass
class MaterialDefinition:
    modelFileName: str
    materialName: str
    materialIndexInModel: int
    manifestTextures: List[Dict[str, Any]]
    references: Optional["MaterialDefinition"] = None
    manifestFileName: str = None
    ntcFileName: str = None

materialDefinitions : List[MaterialDefinition] = []
materialCountsPerModel : Dict[str, int] = {}

def process_gltf_file(inputFileName):
    print(f'\nProcessing {inputFileName}...')

    with open(inputFileName, 'r') as infile:
        gltf = json.load(infile)

    try:    
        materialsNode = gltf['materials']
    except KeyError:
        print('The model doesn\'t have any materials, skipping.')
        return

    materialCountsPerModel[inputFileName] = len(materialsNode)

    try:    
        texturesNode = gltf['textures']
        imagesNode = gltf['images']
        materialsNode = gltf['materials']
    except KeyError:
        print('The model doesn\'t have any textures, skipping.')
        return

    # Go over the materials in the GLTF file, create a manifest file and a compression task for each material.
    for materialIndex, material in enumerate(materialsNode):
        materialName = material.get('name', f'Material')

        print()
        print(f'Material: "{materialName}"')

        gltfDir = os.path.dirname(inputFileName)

        manifestTextures = []

        # Add all supported textures that are present in the material to the manifest.
        for texture in textureTypes:
            add_texture_to_manifest(material, manifestTextures, texture, texturesNode,
                                    imagesNode, gltfDir, manifestDir = args.output)

        if len(manifestTextures) == 0:
            print(f'No textures, skipping.')
            continue

        matdef = MaterialDefinition(
            modelFileName = inputFileName,
            materialName = materialName,
            materialIndexInModel = materialIndex,
            manifestTextures = manifestTextures)
        
        materialDefinitions.append(matdef)

for inputFileName in args.inputFiles:
    process_gltf_file(inputFileName)
print()

materialCountsPerName = {}
def deduplicate_material_name(matdef: MaterialDefinition):
    # Deduplicate material names.
    # Sometimes, multiple glTF materials have the same name, and we need distinct names to associate
    # a single .ntc file with each material. Append _{count} to duplicate names.
    if matdef.materialName in materialCountsPerName:
        # Get the previous count and update the dictionary
        count = materialCountsPerName[matdef.materialName]
        count += 1
        materialCountsPerName[matdef.materialName] = count

        # Print the message, rename the material.
        newMaterialName = f'{matdef.materialName}_{count}'
        print(f'Renaming "{matdef.materialName}" to "{newMaterialName}"')
        matdef.materialName = newMaterialName
    else:
        # This is not a duplicate name yet, store a count of 1.
        materialCountsPerName[matdef.materialName] = 1


tasks : List[ntc.Arguments] = []

for materialIndex in range(len(materialDefinitions)):
    matdef = materialDefinitions[materialIndex]
    for searchIndex in range(materialIndex):
        matdef2 = materialDefinitions[searchIndex]
        if matdef.manifestTextures == matdef2.manifestTextures:
            print(f'Material [{get_file_basename(matdef.modelFileName)} / {matdef.materialName}] uses the same textures as '
                  f'[{get_file_basename(matdef2.modelFileName)} / {matdef2.materialName}], merging.')
            matdef.references = matdef2
            break

    if matdef.references is None:
        deduplicate_material_name(matdef)

        matdef.manifestFileName = os.path.join(args.output, f'{matdef.materialName}.manifest.json')
        matdef.ntcFileName = os.path.join(args.output, f'{matdef.materialName}.ntc')

        with open(matdef.manifestFileName, 'w') as manifestFile:
            # Package the textures into a full manifest
            manifest = { 'textures': matdef.manifestTextures }
            # Save the JSON file
            json.dump(manifest, manifestFile, indent=2)

        if args.skipExisting and os.path.exists(matdef.ntcFileName):
            print(f'Output file {matdef.ntcFileName} exists, skipping.')
            continue

        task = ntc.Arguments(
            tool=args.tool,
            bitsPerPixel=args.bitsPerPixel,
            compress=True,
            decompress=True,
            generateMips=True,
            graphicsApi='vk',
            loadManifest=matdef.manifestFileName,
            maxBitsPerPixel=args.maxBitsPerPixel,
            networkVersion=args.networkVersion,
            optimizeBC=True,
            saveCompressed=matdef.ntcFileName,
            targetPsnr=args.targetPsnr
        )
        
        tasks.append(task)

for inputFileName in args.inputFiles:
    if inputFileName not in materialCountsPerModel:
        continue
    mappingFileName = os.path.join(args.output, f'{get_file_stem(inputFileName)}.ntc-materials.txt')
    with open(mappingFileName, 'w') as mappingFile:
        for materialIndex in range(materialCountsPerModel[inputFileName]):
            found = False
            for matdef in materialDefinitions:
                if matdef.modelFileName == inputFileName and matdef.materialIndexInModel == materialIndex:
                    found = True
                    break
                
            if found:
                if matdef.references is not None:
                    matdef = matdef.references

                print(get_file_basename(matdef.ntcFileName), file = mappingFile)
            else:
                print('*', file = mappingFile)

if args.dryRun:
    sys.exit(0)

maxNtcFilePathLen = max([len(args.saveCompressed) for args in tasks])

def task_ready(task, result, originalTaskCount, tasksCompleted):

    # Delete the manifest file unless instructed to keep it.
    if not args.keepManifests:
        os.unlink(task.loadManifest)

    # Print the status output
    bpp = f'{result.bitsPerPixel:.2f}' if result.bitsPerPixel else 'N/A'
    psnr = f'{result.overallPsnr:.2f}' if result.overallPsnr else 'N/A'
    warning = ' <-- WARNING' if (result.overallPsnr and result.overallPsnr <= 10) else ''
    print(f'[{tasksCompleted:2} of {originalTaskCount:2}] {task.saveCompressed:{maxNtcFilePathLen}}'
            f' : {bpp} bpp, {psnr} dB{warning}')

print()
print('Starting compression...')

terminated = ntc.process_concurrent_tasks(tasks, args.devices, task_ready)

if terminated:
    sys.exit(2)