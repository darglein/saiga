/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ShaderModule.h"

#include "saiga/util/file.h"

#include "GLSL.h"
#include "ShaderLoaderHelper.h"

#if defined(SAIGA_OPENGL_INCLUDED)
#    error OpenGL was included somewhere.
#endif


namespace Saiga
{
namespace Vulkan
{
void ShaderModule::createSPIRV(vk::Device device, vk::ShaderStageFlagBits _stage, const void* data, size_t size)
{
    SAIGA_ASSERT(size % 4 == 0);
    stage        = _stage;
    this->device = device;
    vk::ShaderModuleCreateInfo moduleCreateInfo{};
    moduleCreateInfo.codeSize = size;
    moduleCreateInfo.pCode    = (const uint32_t*)data;
    CHECK_VK(device.createShaderModule(&moduleCreateInfo, nullptr, &module));
}

void ShaderModule::createSPIRV(vk::Device device, vk::ShaderStageFlagBits _stage, const std::vector<uint32_t>& data)
{
    createSPIRV(device, _stage, data.data(), data.size() * sizeof(uint32_t));
}

void ShaderModule::createGLSL(vk::Device device, vk::ShaderStageFlagBits _stage, const std::string& data)
{
    auto spirv = GLSLANG::createFromString(data, _stage);
    createSPIRV(device, _stage, spirv);
}

bool ShaderModule::loadSPIRV(vk::Device device, vk::ShaderStageFlagBits _stage, const std::string& file)
{
    auto data = GLSLANG::loadSPIRV(file);
    if (data.size() == 0) return false;
    createSPIRV(device, _stage, data);
    cout << "Loaded ShaderModule " << file << endl;
    return true;
}

bool ShaderModule::loadGLSL(vk::Device device, vk::ShaderStageFlagBits _stage, const std::string& file,
                            const std::string& injection)
{
    this->file      = file;
    this->injection = injection;
    auto spirv      = GLSLANG::loadGLSL(file, _stage, injection);
    if (spirv.size() == 0) return false;
    createSPIRV(device, _stage, spirv);
    cout << "Loaded ShaderModule " << file << endl;
    return true;
}

bool ShaderModule::load(vk::Device device, const std::string& file, const std::string& injection)
{
    auto ending = ShaderLoadHelper::getEnding(file);
    SAIGA_ASSERT(std::get<0>(ending) != ShaderLoadHelper::ShaderEnding::UNKN);

    if (std::get<0>(ending) == ShaderLoadHelper::ShaderEnding::SPIR)
    {
        ending = ShaderLoadHelper::getEnding(ShaderLoadHelper::stripEnding(file));
        SAIGA_ASSERT(std::get<0>(ending) != ShaderLoadHelper::ShaderEnding::UNKN);
        return loadSPIRV(device, std::get<2>(ending), file);
    }
    else
    {
        return loadGLSL(device, std::get<2>(ending), file, injection);
    }
}

void ShaderModule::destroy()
{
    if (module)
    {
        vkDestroyShaderModule(device, module, nullptr);
        module = nullptr;
    }
}

void ShaderModule::reload()
{
    SAIGA_ASSERT(!file.empty());
    SAIGA_ASSERT(device);

    // Destroy the current module
    destroy();
    cout << "reload shader module." << endl;
    loadGLSL(device, stage, file, injection);
}

bool ShaderModule::valid()
{
    return (bool)module;
}

vk::PipelineShaderStageCreateInfo ShaderModule::createPipelineInfo()
{
    vk::PipelineShaderStageCreateInfo info;
    info.pSpecializationInfo = NULL;
    info.stage               = stage;
    info.pName               = "main";
    info.module              = module;
    return info;
}



}  // namespace Vulkan
}  // namespace Saiga
