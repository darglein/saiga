/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ShaderLoader.h"
#include <fstream>

#include "GLSL.h"

namespace Saiga {
namespace Vulkan {


ShaderLoader shaderLoader;

std::string readTextFile(const char *fileName)
{
    std::string fileContent;
    std::ifstream fileStream(fileName, std::ios::in);
    if (!fileStream.is_open()) {
        printf("File %s not found\n", fileName);
        return "";
    }
    std::string line = "";
    while (!fileStream.eof()) {
        getline(fileStream, line);
        fileContent.append(line + "\n");
    }
    fileStream.close();
    return fileContent;
}

void ShaderLoader::destroy()
{
    for (auto& shaderModule : shaderModules)
    {
//        vkDestroyShaderModule(device, shaderModule, nullptr);
        shaderModule.destroy(device);
    }
}

ShaderModule ShaderLoader::loadModule(const char *fileName, vk::ShaderStageFlagBits stage)
{
    std::ifstream is(fileName, std::ios::binary | std::ios::in | std::ios::ate);

    if (is.is_open())
    {
        size_t size = is.tellg();
        SAIGA_ASSERT(size % 4 == 0);
        SAIGA_ASSERT(size > 0);

        std::vector<uint32_t> spirv(size/4);
        is.seekg(0, std::ios::beg);
//        char* shaderCode = new char[size];
        is.read( (char*)spirv.data(), size);
        is.close();

        ShaderModule shaderModule;
        shaderModule.create(device,stage,spirv);

        return shaderModule;
    }
    else
    {
        std::cerr << "Error: Could not open shader file \"" << fileName << "\"" << std::endl;
        return ShaderModule();
    }
}

ShaderModule ShaderLoader::loadModuleGLSL(const char *fileName, vk::ShaderStageFlagBits stage)
{
    cout << "loadShaderGLSL2 " << fileName << endl;
    std::string shaderSrc = readTextFile(fileName);
    const char *shaderCode = shaderSrc.c_str();


//    ;

    std::vector<unsigned int> vtx_spv = GLSLtoSPIRV(shaderCode,stage);

//    bool retVal = GLSLtoSPV(stage, shaderCode, vtx_spv);
//    SAIGA_ASSERT(retVal);

//    VkShaderModule shaderModule;
//    VkShaderModuleCreateInfo moduleCreateInfo{};
//    moduleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
//    moduleCreateInfo.codeSize = vtx_spv.size() * sizeof(uint32_t);
//    moduleCreateInfo.pCode = (uint32_t*)vtx_spv.data();

//    vkCreateShaderModule(device, &moduleCreateInfo, NULL, &shaderModule);


    ShaderModule shaderModule;
    shaderModule.create(device,stage,vtx_spv);
    return shaderModule;
}

vk::PipelineShaderStageCreateInfo ShaderLoader::loadShader(std::string fileName, vk::ShaderStageFlagBits stage)
{
    auto module = Saiga::Vulkan::shaderLoader.loadModule(fileName.c_str(),stage);
    shaderModules.push_back(module);
    return module.createPipelineInfo();
}

vk::PipelineShaderStageCreateInfo ShaderLoader::loadShaderGLSL(std::string fileName, vk::ShaderStageFlagBits stage)
{
    auto module = Saiga::Vulkan::shaderLoader.loadModuleGLSL(fileName.c_str(),stage);
    shaderModules.push_back(module);
    return module.createPipelineInfo();
}

}
}
