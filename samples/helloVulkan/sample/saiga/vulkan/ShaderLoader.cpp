/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ShaderLoader.h"
#include <fstream>


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
        vkDestroyShaderModule(device, shaderModule, nullptr);
    }
}

VkShaderModule ShaderLoader::loadShader(const char *fileName)
{
    std::ifstream is(fileName, std::ios::binary | std::ios::in | std::ios::ate);

    if (is.is_open())
    {
        size_t size = is.tellg();
        is.seekg(0, std::ios::beg);
        char* shaderCode = new char[size];
        is.read(shaderCode, size);
        is.close();

        assert(size > 0);

        VkShaderModule shaderModule;
        VkShaderModuleCreateInfo moduleCreateInfo{};
        moduleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        moduleCreateInfo.codeSize = size;
        moduleCreateInfo.pCode = (uint32_t*)shaderCode;

        vkCreateShaderModule(device, &moduleCreateInfo, NULL, &shaderModule);

        delete[] shaderCode;

        return shaderModule;
    }
    else
    {
        std::cerr << "Error: Could not open shader file \"" << fileName << "\"" << std::endl;
        return VK_NULL_HANDLE;
    }
}

VkShaderModule ShaderLoader::loadShaderGLSL(const char *fileName, VkShaderStageFlagBits stage)
{
    std::string shaderSrc = readTextFile(fileName);
    const char *shaderCode = shaderSrc.c_str();
    size_t size = strlen(shaderCode);
    assert(size > 0);

    VkShaderModule shaderModule;
    VkShaderModuleCreateInfo moduleCreateInfo;
    moduleCreateInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    moduleCreateInfo.pNext = NULL;
    moduleCreateInfo.codeSize = 3 * sizeof(uint32_t) + size + 1;
    moduleCreateInfo.pCode = (uint32_t*)malloc(moduleCreateInfo.codeSize);
    moduleCreateInfo.flags = 0;

    // Magic SPV number
    ((uint32_t *)moduleCreateInfo.pCode)[0] = 0x07230203;
    ((uint32_t *)moduleCreateInfo.pCode)[1] = 0;
    ((uint32_t *)moduleCreateInfo.pCode)[2] = stage;
    memcpy(((uint32_t *)moduleCreateInfo.pCode + 3), shaderCode, size + 1);

    vkCreateShaderModule(device, &moduleCreateInfo, NULL, &shaderModule);

    return shaderModule;
}

VkPipelineShaderStageCreateInfo ShaderLoader::loadShader(std::string fileName, VkShaderStageFlagBits stage)
{
    VkPipelineShaderStageCreateInfo shaderStage = {};
    shaderStage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    shaderStage.stage = stage;
#if defined(VK_USE_PLATFORM_ANDROID_KHR)
    shaderStage.module = vks::tools::loadShader(androidApp->activity->assetManager, fileName.c_str(), device);
#else
    shaderStage.module = Saiga::Vulkan::shaderLoader.loadShader(fileName.c_str());
#endif
    shaderStage.pName = "main"; // todo : make param
    assert(shaderStage.module != VK_NULL_HANDLE);
    shaderModules.push_back(shaderStage.module);
    return shaderStage;
}

}
}
