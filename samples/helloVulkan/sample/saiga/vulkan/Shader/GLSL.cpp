/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Shader.h"
#include "saiga/vulkan/vulkanHelper.h"
#include "saiga/util/file.h"
#include "SPIRV/GlslangToSpv.h"

namespace Saiga {
namespace Vulkan {


void init_resources(TBuiltInResource &Resources) {
    Resources.maxLights = 32;
    Resources.maxClipPlanes = 6;
    Resources.maxTextureUnits = 32;
    Resources.maxTextureCoords = 32;
    Resources.maxVertexAttribs = 64;
    Resources.maxVertexUniformComponents = 4096;
    Resources.maxVaryingFloats = 64;
    Resources.maxVertexTextureImageUnits = 32;
    Resources.maxCombinedTextureImageUnits = 80;
    Resources.maxTextureImageUnits = 32;
    Resources.maxFragmentUniformComponents = 4096;
    Resources.maxDrawBuffers = 32;
    Resources.maxVertexUniformVectors = 128;
    Resources.maxVaryingVectors = 8;
    Resources.maxFragmentUniformVectors = 16;
    Resources.maxVertexOutputVectors = 16;
    Resources.maxFragmentInputVectors = 15;
    Resources.minProgramTexelOffset = -8;
    Resources.maxProgramTexelOffset = 7;
    Resources.maxClipDistances = 8;
    Resources.maxComputeWorkGroupCountX = 65535;
    Resources.maxComputeWorkGroupCountY = 65535;
    Resources.maxComputeWorkGroupCountZ = 65535;
    Resources.maxComputeWorkGroupSizeX = 1024;
    Resources.maxComputeWorkGroupSizeY = 1024;
    Resources.maxComputeWorkGroupSizeZ = 64;
    Resources.maxComputeUniformComponents = 1024;
    Resources.maxComputeTextureImageUnits = 16;
    Resources.maxComputeImageUniforms = 8;
    Resources.maxComputeAtomicCounters = 8;
    Resources.maxComputeAtomicCounterBuffers = 1;
    Resources.maxVaryingComponents = 60;
    Resources.maxVertexOutputComponents = 64;
    Resources.maxGeometryInputComponents = 64;
    Resources.maxGeometryOutputComponents = 128;
    Resources.maxFragmentInputComponents = 128;
    Resources.maxImageUnits = 8;
    Resources.maxCombinedImageUnitsAndFragmentOutputs = 8;
    Resources.maxCombinedShaderOutputResources = 8;
    Resources.maxImageSamples = 0;
    Resources.maxVertexImageUniforms = 0;
    Resources.maxTessControlImageUniforms = 0;
    Resources.maxTessEvaluationImageUniforms = 0;
    Resources.maxGeometryImageUniforms = 0;
    Resources.maxFragmentImageUniforms = 8;
    Resources.maxCombinedImageUniforms = 8;
    Resources.maxGeometryTextureImageUnits = 16;
    Resources.maxGeometryOutputVertices = 256;
    Resources.maxGeometryTotalOutputComponents = 1024;
    Resources.maxGeometryUniformComponents = 1024;
    Resources.maxGeometryVaryingComponents = 64;
    Resources.maxTessControlInputComponents = 128;
    Resources.maxTessControlOutputComponents = 128;
    Resources.maxTessControlTextureImageUnits = 16;
    Resources.maxTessControlUniformComponents = 1024;
    Resources.maxTessControlTotalOutputComponents = 4096;
    Resources.maxTessEvaluationInputComponents = 128;
    Resources.maxTessEvaluationOutputComponents = 128;
    Resources.maxTessEvaluationTextureImageUnits = 16;
    Resources.maxTessEvaluationUniformComponents = 1024;
    Resources.maxTessPatchComponents = 120;
    Resources.maxPatchVertices = 32;
    Resources.maxTessGenLevel = 64;
    Resources.maxViewports = 16;
    Resources.maxVertexAtomicCounters = 0;
    Resources.maxTessControlAtomicCounters = 0;
    Resources.maxTessEvaluationAtomicCounters = 0;
    Resources.maxGeometryAtomicCounters = 0;
    Resources.maxFragmentAtomicCounters = 8;
    Resources.maxCombinedAtomicCounters = 8;
    Resources.maxAtomicCounterBindings = 1;
    Resources.maxVertexAtomicCounterBuffers = 0;
    Resources.maxTessControlAtomicCounterBuffers = 0;
    Resources.maxTessEvaluationAtomicCounterBuffers = 0;
    Resources.maxGeometryAtomicCounterBuffers = 0;
    Resources.maxFragmentAtomicCounterBuffers = 1;
    Resources.maxCombinedAtomicCounterBuffers = 1;
    Resources.maxAtomicCounterBufferSize = 16384;
    Resources.maxTransformFeedbackBuffers = 4;
    Resources.maxTransformFeedbackInterleavedComponents = 64;
    Resources.maxCullDistances = 8;
    Resources.maxCombinedClipAndCullDistances = 8;
    Resources.maxSamples = 4;
    Resources.limits.nonInductiveForLoops = 1;
    Resources.limits.whileLoops = 1;
    Resources.limits.doWhileLoops = 1;
    Resources.limits.generalUniformIndexing = 1;
    Resources.limits.generalAttributeMatrixVectorIndexing = 1;
    Resources.limits.generalVaryingIndexing = 1;
    Resources.limits.generalSamplerIndexing = 1;
    Resources.limits.generalVariableIndexing = 1;
    Resources.limits.generalConstantMatrixVectorIndexing = 1;
}
EShLanguage FindLanguage(const vk::ShaderStageFlagBits shader_type) {
    switch (shader_type) {
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

struct MyIncluder : public  glslang::TShader::Includer
{
    std::vector<std::string> data;

    virtual IncludeResult* includeSystem(const char* headerName,
                                        const char* includerName,
                                        size_t inclusionDepth)
    {
        cout << "include request " << headerName << " " << includerName << " " << inclusionDepth << endl;


        data.push_back(File::loadFileString(headerName));

        IncludeResult* result = new IncludeResult(headerName,data.back().data(),data.back().size(),nullptr);

        return result;
    }

    virtual IncludeResult* includeLocal(const char* headerName,
                                        const char* includerName,
                                        size_t inclusionDepth)
    {

        return includeSystem(headerName,includerName,inclusionDepth);
    }

 virtual void releaseInclude(IncludeResult*) override { }
};

std::vector<uint32_t> GLSLtoSPIRV(const std::string& shaderString,
        const vk::ShaderStageFlagBits shader_type
        )
{

    std::vector<uint32_t> spirv;

    glslang::InitializeProcess();

    EShLanguage stage = FindLanguage(shader_type);
    glslang::TShader shader(stage);
    glslang::TProgram program;
    const char *shaderStrings[1];
    TBuiltInResource Resources;
    init_resources(Resources);

    // Enable SPIR-V and Vulkan rules when parsing GLSL
    EShMessages messages = (EShMessages)(EShMsgSpvRules | EShMsgVulkanRules);

    shaderStrings[0] = shaderString.c_str();
    shader.setStrings(shaderStrings, 1);

      MyIncluder includer;

    if (!shader.parse(&Resources, 100, false, messages,includer)) {
        puts(shader.getInfoLog());
        puts(shader.getInfoDebugLog());
        return spirv;  // something didn't work
    }

    program.addShader(&shader);

    //
    // Program-level processing...
    //

    if (!program.link(messages)) {
        puts(shader.getInfoLog());
        puts(shader.getInfoDebugLog());
        fflush(stdout);
        return spirv;
    }

    glslang::GlslangToSpv(*program.getIntermediate(stage), spirv);



    glslang::FinalizeProcess();

    return spirv;
}

}
}
