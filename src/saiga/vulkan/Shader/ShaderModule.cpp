/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ShaderModule.h"

#include "saiga/core/util/file.h"
#include "saiga/core/util/fileChecker.h"

#include "GLSL.h"
#include "ShaderLoaderHelper.h"

#include <iostream>
#include <utility>

#if defined(SAIGA_OPENGL_INCLUDED)
#    error OpenGL was included somewhere.
#endif


namespace Saiga
{
namespace Vulkan
{
ShaderModule::ShaderModule(ShaderModule&& other) noexcept
{
    *this = std::move(other);
}

ShaderModule& ShaderModule::operator=(ShaderModule&& other) noexcept
{
    if (this != &other)
    {
        destroy();
        stage        = other.stage;
        module       = std::exchange(other.module, nullptr);
        device       = other.device;
        file         = std::move(other.file);
        injection    = std::move(other.injection);
        isSpirv      = other.isSpirv;
        dependencies = std::move(other.dependencies);
        lastWrites   = std::move(other.lastWrites);
        lastCheck    = other.lastCheck;
    }
    return *this;
}

void ShaderModule::createModule(vk::Device _device, vk::ShaderStageFlagBits _stage, const void* data, size_t size)
{
    SAIGA_ASSERT(size % 4 == 0);
    destroy();
    stage  = _stage;
    device = _device;
    if (size == 0)
    {
        // Compilation failed: leave the module invalid (see valid()) instead of creating an empty one.
        return;
    }
    vk::ShaderModuleCreateInfo moduleCreateInfo{};
    moduleCreateInfo.codeSize = size;
    moduleCreateInfo.pCode    = (const uint32_t*)data;
    CHECK_VK(device.createShaderModule(&moduleCreateInfo, nullptr, &module));
}

void ShaderModule::createSPIRV(vk::Device device, vk::ShaderStageFlagBits _stage, const void* data, size_t size)
{
    // Not loaded from a file -> no reloading
    file.clear();
    injection.clear();
    dependencies.clear();
    lastWrites.clear();
    isSpirv = true;
    createModule(device, _stage, data, size);
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
    // SPIR-V files are not reloaded
    createSPIRV(device, _stage, data);
    if (data.empty()) return false;
    std::cout << "Loaded ShaderModule " << file << std::endl;
    return true;
}

bool ShaderModule::loadGLSL(vk::Device _device, vk::ShaderStageFlagBits _stage, const std::string& _file,
                            const std::string& _injection)
{
    destroy();
    device    = _device;
    stage     = _stage;
    isSpirv   = false;
    injection = _injection;
    file      = SearchPathes::shader(_file).string();
    if (file.empty())
    {
        std::cout << "Could not find " << _file << std::endl;
        dependencies.clear();
        lastWrites.clear();
        return false;
    }

    // The write times before compiling: a change during the compilation triggers another reload.
    dependencies = {file};
    lastWrites   = getWriteTimes();
    lastCheck    = std::chrono::steady_clock::now();

    std::vector<std::string> includes;
    auto spirv = GLSLANG::loadGLSL(file, _stage, injection, &includes);
    dependencies.insert(dependencies.end(), includes.begin(), includes.end());
    lastWrites.resize(1);
    auto includeTimes = getWriteTimes();
    lastWrites.insert(lastWrites.end(), includeTimes.begin() + 1, includeTimes.end());

    if (spirv.empty()) return false;
    createModule(device, _stage, spirv.data(), spirv.size() * sizeof(uint32_t));
    std::cout << "Loaded ShaderModule " << file << std::endl;
    return true;
}

bool ShaderModule::load(vk::Device device, const std::string& file, const std::string& injection)
{
    auto ending = ShaderLoadHelper::getEnding(file);
    SAIGA_ASSERT(std::get<0>(ending) != ShaderLoadHelper::ShaderEnding::UNKN, file);

    if (std::get<0>(ending) == ShaderLoadHelper::ShaderEnding::SPIR)
    {
        ending = ShaderLoadHelper::getEnding(ShaderLoadHelper::stripEnding(file));
        SAIGA_ASSERT(std::get<0>(ending) != ShaderLoadHelper::ShaderEnding::UNKN, file);
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
        device.destroyShaderModule(module);
        module = nullptr;
    }
}

bool ShaderModule::reload()
{
    if (file.empty() || isSpirv || !device) return false;

    // Compile first and keep the current module if it fails, so a pipeline can still be (re)created from it.
    dependencies.resize(1);
    lastWrites = getWriteTimes();
    lastCheck  = std::chrono::steady_clock::now();

    std::vector<std::string> includes;
    auto spirv = GLSLANG::loadGLSL(file, stage, injection, &includes);
    dependencies.insert(dependencies.end(), includes.begin(), includes.end());
    auto times = getWriteTimes();
    lastWrites.insert(lastWrites.end(), times.begin() + 1, times.end());

    if (spirv.empty())
    {
        std::cout << "Reloading " << file << " failed. Keeping the previous shader module." << std::endl;
        return false;
    }
    createModule(device, stage, spirv.data(), spirv.size() * sizeof(uint32_t));
    std::cout << "Reloaded ShaderModule " << file << std::endl;
    return true;
}

bool ShaderModule::valid()
{
    return (bool)module;
}

std::vector<std::filesystem::file_time_type> ShaderModule::getWriteTimes() const
{
    std::vector<std::filesystem::file_time_type> times(dependencies.size());
    for (size_t i = 0; i < dependencies.size(); ++i)
    {
        std::error_code ec;
        auto t = std::filesystem::last_write_time(dependencies[i], ec);
        // A missing file (e.g. while an editor saves it) is treated as unchanged: see autoReload().
        times[i] = ec ? std::filesystem::file_time_type::min() : t;
    }
    return times;
}

bool ShaderModule::autoReload()
{
    if (file.empty() || isSpirv || dependencies.empty()) return false;

    // Called on every bind: only look at the file system every 0.5 s.
    auto now = std::chrono::steady_clock::now();
    if (now - lastCheck < std::chrono::milliseconds(500)) return false;
    lastCheck = now;

    auto times = getWriteTimes();
    // The shader file itself is missing (e.g. while an editor saves it): try again later. A missing include is a
    // change (e.g. it was renamed and the #include updated): reload() rebuilds the dependency list, also if the
    // compilation fails, so this does not repeat.
    if (times[0] == std::filesystem::file_time_type::min()) return false;
    if (times == lastWrites) return false;

    std::cout << "File Update detected. Reloading " << file << std::endl;
    return reload();
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
