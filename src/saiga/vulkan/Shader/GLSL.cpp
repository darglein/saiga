/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "GLSL.h"

#include "saiga/core/util/file.h"
#include "saiga/core/util/fileChecker.h"
#include "saiga/core/util/tostring.h"

#include "SPIRV/GlslangToSpv.h"
#include "SPIRV/spirv.hpp"

#include <iostream>

namespace Saiga
{
namespace Vulkan
{
namespace GLSLANG
{
static TBuiltInResource Resources;


static EShLanguage FindLanguage(const vk::ShaderStageFlagBits shader_type)
{
    switch (shader_type)
    {
        case vk::ShaderStageFlagBits::eVertex:
            return EShLangVertex;

        case vk::ShaderStageFlagBits::eTessellationControl:
            return EShLangTessControl;

        case vk::ShaderStageFlagBits::eTessellationEvaluation:
            return EShLangTessEvaluation;

        case vk::ShaderStageFlagBits::eGeometry:
            return EShLangGeometry;

        case vk::ShaderStageFlagBits::eFragment:
            return EShLangFragment;

        case vk::ShaderStageFlagBits::eCompute:
            return EShLangCompute;

        default:
            SAIGA_ASSERT(0);
            return EShLangVertex;
    }
}


struct MyIncluder : public glslang::TShader::Includer
{
    std::vector<std::string> data;
    std::string baseFile;
    std::vector<IncludeResult*> results;

    virtual ~MyIncluder()
    {
        for (auto r : results) delete r;
    }

    virtual IncludeResult* includeSystem(const char* headerName, const char* includerName,
                                         size_t inclusionDepth) override
    {
        // be save from std::endless loops
        if (inclusionDepth > 100)
        {
            return nullptr;
        }
        std::string base = std::string(includerName).size() > 0 ? std::string(includerName) : baseFile;
        //        std::cout << "include request '" << headerName << "' '" << includerName << "' " << inclusionDepth <<
        //        std::endl; std::cout << "base " << base << std::endl;

        auto includeFileName = SearchPathes::shader.getRelative(base, headerName);

        if (includeFileName == "")
        {
            //            std::cout << "relative include not found" << std::endl;
            return nullptr;
        }
        else
        {
            //            std::cout << "found " << includeFileName << std::endl;
        }

        data.push_back(File::loadFileString(includeFileName));

        IncludeResult* result = new IncludeResult(includeFileName, data.back().data(), data.back().size(), nullptr);
        results.push_back(result);

        return result;
    }

    virtual IncludeResult* includeLocal(const char* headerName, const char* includerName,
                                        size_t inclusionDepth) override
    {
        return includeSystem(headerName, includerName, inclusionDepth);
    }

    virtual void releaseInclude(IncludeResult*) override {}
};

std::vector<uint32_t> createFromString(const std::string& shaderString, const vk::ShaderStageFlagBits shader_type)
{
    glslang::TShader shader(FindLanguage(shader_type));

    const char* shaderStrings[1] = {shaderString.c_str()};
    shader.setStrings(shaderStrings, 1);


    // Enable SPIR-V and Vulkan rules when parsing GLSL
    EShMessages messages = (EShMessages)(EShMsgSpvRules | EShMsgVulkanRules);

    if (!shader.parse(&Resources, 100, false, messages))
    {
        std::cout << shader.getInfoLog() << std::endl;
        std::cout << shader.getInfoDebugLog() << std::endl;
        return {};  // something didn't work
    }

    std::vector<uint32_t> spirv;
    glslang::GlslangToSpv(*shader.getIntermediate(), spirv);
    return spirv;
}

std::vector<uint32_t> loadGLSL(const std::string& _file, const vk::ShaderStageFlagBits shader_type,
                               const std::string& injection)
{
    auto file = SearchPathes::shader(_file);

    if (file == "")
    {
        std::cout << "Could not find " << _file << std::endl;
        SAIGA_ASSERT(0);
    }

    auto shaderString = Saiga::File::loadFileString(file);

    glslang::TShader shader(FindLanguage(shader_type));


    MyIncluder includer;
    includer.baseFile = file;

    // Enable SPIR-V and Vulkan rules when parsing GLSL
    EShMessages messages = (EShMessages)(EShMsgSpvRules | EShMsgVulkanRules);

    if (!injection.empty()) addInjectionAfterVersion(shaderString, injection);


    const char* shaderStrings[1] = {shaderString.c_str()};
    shader.setStrings(shaderStrings, 1);

    if (!shader.parse(&Resources, 100, false, messages, includer))
    {
        std::cout << "Error in " << _file << std::endl;
        std::cout << shader.getInfoLog() << std::endl;
        std::cout << shader.getInfoDebugLog() << std::endl;
        return {};  // something didn't work
    }

    std::vector<uint32_t> spirv;
    glslang::GlslangToSpv(*shader.getIntermediate(), spirv);

    //    std::vector<uint32_t> spvinjection = {0x20011, 12 };
    //    spirv.insert(spirv.begin()+7,spvinjection.begin(),spvinjection.end());

    //    for(int i = 0 ;i < 20; ++i)
    //    {
    //        std::cout << std::hex << spirv[i] << std::endl;
    //    }
    return spirv;
}

std::vector<uint32_t> loadSPIRV(const std::string& file)
{
    auto file2 = SearchPathes::shader(file);
    auto data  = Saiga::File::loadFileBinary(file2);
    SAIGA_ASSERT(data.size() % 4 == 0);
    std::vector<uint32_t> spirv(data.size() / 4);
    memcpy(spirv.data(), data.data(), data.size());
    return spirv;
}

void addInjectionAfterVersion(std::string& shaderString, const std::string& injections)
{
    // search for version line
    size_t pos = shaderString.find("#version");
    SAIGA_ASSERT(pos != std::string::npos);
    // go to \n
    bool found = false;
    for (; pos < shaderString.size(); ++pos)
    {
        if (shaderString[pos] == '\n')
        {
            found = true;
            break;
        }
    }
    SAIGA_ASSERT(found);
    shaderString.insert(shaderString.begin() + pos, '\n');
    shaderString.insert(shaderString.begin() + pos + 1, injections.begin(), injections.end());
}


static void init_resources(TBuiltInResource& Resources)
{
    Resources.maxLights                                   = 32;
    Resources.maxClipPlanes                               = 6;
    Resources.maxTextureUnits                             = 32;
    Resources.maxTextureCoords                            = 32;
    Resources.maxVertexAttribs                            = 64;
    Resources.maxVertexUniformComponents                  = 4096;
    Resources.maxVaryingFloats                            = 64;
    Resources.maxVertexTextureImageUnits                  = 32;
    Resources.maxCombinedTextureImageUnits                = 80;
    Resources.maxTextureImageUnits                        = 32;
    Resources.maxFragmentUniformComponents                = 4096;
    Resources.maxDrawBuffers                              = 32;
    Resources.maxVertexUniformVectors                     = 128;
    Resources.maxVaryingVectors                           = 8;
    Resources.maxFragmentUniformVectors                   = 16;
    Resources.maxVertexOutputVectors                      = 16;
    Resources.maxFragmentInputVectors                     = 15;
    Resources.minProgramTexelOffset                       = -8;
    Resources.maxProgramTexelOffset                       = 7;
    Resources.maxClipDistances                            = 8;
    Resources.maxComputeWorkGroupCountX                   = 65535;
    Resources.maxComputeWorkGroupCountY                   = 65535;
    Resources.maxComputeWorkGroupCountZ                   = 65535;
    Resources.maxComputeWorkGroupSizeX                    = 1024;
    Resources.maxComputeWorkGroupSizeY                    = 1024;
    Resources.maxComputeWorkGroupSizeZ                    = 64;
    Resources.maxComputeUniformComponents                 = 1024;
    Resources.maxComputeTextureImageUnits                 = 16;
    Resources.maxComputeImageUniforms                     = 8;
    Resources.maxComputeAtomicCounters                    = 8;
    Resources.maxComputeAtomicCounterBuffers              = 1;
    Resources.maxVaryingComponents                        = 60;
    Resources.maxVertexOutputComponents                   = 64;
    Resources.maxGeometryInputComponents                  = 64;
    Resources.maxGeometryOutputComponents                 = 128;
    Resources.maxFragmentInputComponents                  = 128;
    Resources.maxImageUnits                               = 8;
    Resources.maxCombinedImageUnitsAndFragmentOutputs     = 8;
    Resources.maxCombinedShaderOutputResources            = 8;
    Resources.maxImageSamples                             = 0;
    Resources.maxVertexImageUniforms                      = 0;
    Resources.maxTessControlImageUniforms                 = 0;
    Resources.maxTessEvaluationImageUniforms              = 0;
    Resources.maxGeometryImageUniforms                    = 0;
    Resources.maxFragmentImageUniforms                    = 8;
    Resources.maxCombinedImageUniforms                    = 8;
    Resources.maxGeometryTextureImageUnits                = 16;
    Resources.maxGeometryOutputVertices                   = 256;
    Resources.maxGeometryTotalOutputComponents            = 1024;
    Resources.maxGeometryUniformComponents                = 1024;
    Resources.maxGeometryVaryingComponents                = 64;
    Resources.maxTessControlInputComponents               = 128;
    Resources.maxTessControlOutputComponents              = 128;
    Resources.maxTessControlTextureImageUnits             = 16;
    Resources.maxTessControlUniformComponents             = 1024;
    Resources.maxTessControlTotalOutputComponents         = 4096;
    Resources.maxTessEvaluationInputComponents            = 128;
    Resources.maxTessEvaluationOutputComponents           = 128;
    Resources.maxTessEvaluationTextureImageUnits          = 16;
    Resources.maxTessEvaluationUniformComponents          = 1024;
    Resources.maxTessPatchComponents                      = 120;
    Resources.maxPatchVertices                            = 32;
    Resources.maxTessGenLevel                             = 64;
    Resources.maxViewports                                = 16;
    Resources.maxVertexAtomicCounters                     = 0;
    Resources.maxTessControlAtomicCounters                = 0;
    Resources.maxTessEvaluationAtomicCounters             = 0;
    Resources.maxGeometryAtomicCounters                   = 0;
    Resources.maxFragmentAtomicCounters                   = 8;
    Resources.maxCombinedAtomicCounters                   = 8;
    Resources.maxAtomicCounterBindings                    = 1;
    Resources.maxVertexAtomicCounterBuffers               = 0;
    Resources.maxTessControlAtomicCounterBuffers          = 0;
    Resources.maxTessEvaluationAtomicCounterBuffers       = 0;
    Resources.maxGeometryAtomicCounterBuffers             = 0;
    Resources.maxFragmentAtomicCounterBuffers             = 1;
    Resources.maxCombinedAtomicCounterBuffers             = 1;
    Resources.maxAtomicCounterBufferSize                  = 16384;
    Resources.maxTransformFeedbackBuffers                 = 4;
    Resources.maxTransformFeedbackInterleavedComponents   = 64;
    Resources.maxCullDistances                            = 8;
    Resources.maxCombinedClipAndCullDistances             = 8;
    Resources.maxSamples                                  = 4;
    Resources.limits.nonInductiveForLoops                 = 1;
    Resources.limits.whileLoops                           = 1;
    Resources.limits.doWhileLoops                         = 1;
    Resources.limits.generalUniformIndexing               = 1;
    Resources.limits.generalAttributeMatrixVectorIndexing = 1;
    Resources.limits.generalVaryingIndexing               = 1;
    Resources.limits.generalSamplerIndexing               = 1;
    Resources.limits.generalVariableIndexing              = 1;
    Resources.limits.generalConstantMatrixVectorIndexing  = 1;
}
void init()
{
    glslang::InitializeProcess();
    init_resources(Resources);
}

void quit()
{
    glslang::FinalizeProcess();
}



}  // namespace GLSLANG
}  // namespace Vulkan
}  // namespace Saiga
