/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "GLSL.h"

#include "saiga/core/util/file.h"
#include "saiga/core/util/fileChecker.h"
#include "saiga/core/util/tostring.h"

// glslang >= 11 installs its headers as glslang/SPIRV/..., older versions as SPIRV/...
#if __has_include("glslang/SPIRV/GlslangToSpv.h")
#    include "glslang/SPIRV/GlslangToSpv.h"
#else
#    include "SPIRV/GlslangToSpv.h"
#endif
#include "glslang/Public/ShaderLang.h"
#if __has_include("glslang/build_info.h")
#    include "glslang/build_info.h"
#endif

#include <cstring>
#include <deque>
#include <iostream>

namespace Saiga
{
namespace Vulkan
{
namespace GLSLANG
{
// The resource limits of the compiled shaders. Filled once, see GetResources() below.
static const TBuiltInResource& GetResources();


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


// Links the (single stage) program, which runs glslang's link-time checks, e.g. a missing main, and generates SPIR-V.
// Returns an empty vector on failure.
static std::vector<uint32_t> LinkToSpirv(glslang::TShader& shader, EShLanguage stage, EShMessages messages)
{
    glslang::TProgram program;
    program.addShader(&shader);
    if (!program.link(messages))
    {
        std::cout << program.getInfoLog() << std::endl;
        std::cout << program.getInfoDebugLog() << std::endl;
        return {};
    }

    std::vector<uint32_t> spirv;
    spv::SpvBuildLogger logger;
    glslang::GlslangToSpv(*program.getIntermediate(stage), spirv, &logger);
    auto messages_str = logger.getAllMessages();
    if (!messages_str.empty()) std::cout << messages_str << std::endl;
    return spirv;
}

struct MyIncluder : public glslang::TShader::Includer
{
    // A deque never moves its elements, so the pointers handed to glslang stay valid when more includes are loaded.
    // (A vector would move the strings, which invalidates the data of short strings stored inline.)
    std::deque<std::string> data;
    std::string baseFile;
    std::vector<IncludeResult*> results;
    // All included files (for the hot reload of a shader when one of its includes changes)
    std::vector<std::string> includedFiles;

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

        auto includeFileName = SearchPathes::shader.getRelative(base, headerName).string();

        if (includeFileName.empty())
        {
            return nullptr;
        }

        includedFiles.push_back(includeFileName);
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

    if (!shader.parse(&GetResources(), 100, false, messages))
    {
        std::cout << shader.getInfoLog() << std::endl;
        std::cout << shader.getInfoDebugLog() << std::endl;
        return {};  // something didn't work
    }

    return LinkToSpirv(shader, FindLanguage(shader_type), messages);
}

std::vector<uint32_t> loadGLSL(const std::string& _file, const vk::ShaderStageFlagBits shader_type,
                               const std::string& injection, std::vector<std::string>* includedFiles)
{
    if (includedFiles) includedFiles->clear();
    auto file = SearchPathes::shader(_file).string();

    // Not an assert: during a hot reload the file may be missing temporarily (e.g. an editor saving by rename).
    if (file.empty())
    {
        std::cout << "Could not find " << _file << std::endl;
        return {};
    }

    auto shaderString = Saiga::File::loadFileString(file);
    if (shaderString.empty())
    {
        std::cout << "Could not load " << file << std::endl;
        return {};
    }

    glslang::TShader shader(FindLanguage(shader_type));


    MyIncluder includer;
    includer.baseFile = file;

    // Enable SPIR-V and Vulkan rules when parsing GLSL
    EShMessages messages = (EShMessages)(EShMsgSpvRules | EShMsgVulkanRules);

    if (!injection.empty() && !addInjectionAfterVersion(shaderString, injection))
    {
        std::cout << "Error in " << _file << ": no #version line for the injection" << std::endl;
        return {};
    }


    const char* shaderStrings[1] = {shaderString.c_str()};
    shader.setStrings(shaderStrings, 1);

    bool parsed = shader.parse(&GetResources(), 100, false, messages, includer);
    if (includedFiles) *includedFiles = includer.includedFiles;
    if (!parsed)
    {
        std::cout << "Error in " << _file << std::endl;
        std::cout << shader.getInfoLog() << std::endl;
        std::cout << shader.getInfoDebugLog() << std::endl;
        return {};  // something didn't work
    }

    auto spirv = LinkToSpirv(shader, FindLanguage(shader_type), messages);
    if (spirv.empty()) std::cout << "Error in " << _file << std::endl;
    return spirv;
}

std::vector<uint32_t> loadSPIRV(const std::string& file)
{
    auto file2 = SearchPathes::shader(file);
    if (file2.empty())
    {
        std::cout << "Could not find " << file << std::endl;
        return {};
    }
    auto data = Saiga::File::loadFileBinary(file2);
    if (data.size() % 4 != 0)
    {
        std::cout << "Invalid SPIR-V file (size not a multiple of 4) " << file << std::endl;
        return {};
    }
    std::vector<uint32_t> spirv(data.size() / 4);
    memcpy(spirv.data(), data.data(), data.size());
    return spirv;
}

bool addInjectionAfterVersion(std::string& shaderString, const std::string& injections)
{
    // search for version line
    size_t pos = shaderString.find("#version");
    if (pos == std::string::npos) return false;
    // go to the end of the line
    pos = shaderString.find('\n', pos);
    if (pos == std::string::npos) return false;
    shaderString.insert(pos, "\n" + injections);
    return true;
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

    // Limits added in later glslang versions (the values of glslang's GetDefaultResources()). They are 0 otherwise,
    // which makes e.g. mesh shaders or dual source blending fail to compile.
#if defined(GLSLANG_VERSION_MAJOR) && GLSLANG_VERSION_MAJOR >= 12
    Resources.maxMeshOutputVerticesNV     = 256;
    Resources.maxMeshOutputPrimitivesNV   = 512;
    Resources.maxMeshWorkGroupSizeX_NV    = 32;
    Resources.maxMeshWorkGroupSizeY_NV    = 1;
    Resources.maxMeshWorkGroupSizeZ_NV    = 1;
    Resources.maxTaskWorkGroupSizeX_NV    = 32;
    Resources.maxTaskWorkGroupSizeY_NV    = 1;
    Resources.maxTaskWorkGroupSizeZ_NV    = 1;
    Resources.maxMeshViewCountNV          = 4;
    Resources.maxMeshOutputVerticesEXT    = 256;
    Resources.maxMeshOutputPrimitivesEXT  = 256;
    Resources.maxMeshWorkGroupSizeX_EXT   = 128;
    Resources.maxMeshWorkGroupSizeY_EXT   = 128;
    Resources.maxMeshWorkGroupSizeZ_EXT   = 128;
    Resources.maxTaskWorkGroupSizeX_EXT   = 128;
    Resources.maxTaskWorkGroupSizeY_EXT   = 128;
    Resources.maxTaskWorkGroupSizeZ_EXT   = 128;
    Resources.maxMeshViewCountEXT         = 4;
    Resources.maxDualSourceDrawBuffersEXT = 1;
#endif
}

static const TBuiltInResource& GetResources()
{
    // Value-initialized: fields unknown to this code stay 0.
    static const TBuiltInResource resources = []()
    {
        TBuiltInResource r = {};
        init_resources(r);
        return r;
    }();
    return resources;
}

void init()
{
    glslang::InitializeProcess();
}

void quit()
{
    glslang::FinalizeProcess();
}



}  // namespace GLSLANG
}  // namespace Vulkan
}  // namespace Saiga
