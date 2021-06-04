/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once


#include "saiga/vulkan/Shader/ShaderModule.h"
#include "saiga/vulkan/Shader/ShaderPipeline.h"
#include "saiga/vulkan/svulkan.h"

namespace Saiga
{
namespace Vulkan
{
namespace ShaderLoadHelper
{
enum class ShaderEnding
{
    VERT,
    TESC,
    TESE,
    GEOM,
    FRAG,
    COMP,
    SPIR,
    UNKN  // unknown ending
};

using EndingType = std::tuple<ShaderEnding, std::string, vk::ShaderStageFlagBits>;

const std::array<EndingType, 8> fileEndings = {
    {{ShaderEnding::VERT, "vert", vk::ShaderStageFlagBits::eVertex},
     {ShaderEnding::TESC, "tesc", vk::ShaderStageFlagBits::eTessellationControl},
     {ShaderEnding::TESE, "tese", vk::ShaderStageFlagBits::eTessellationEvaluation},
     {ShaderEnding::GEOM, "geom", vk::ShaderStageFlagBits::eGeometry},
     {ShaderEnding::FRAG, "frag", vk::ShaderStageFlagBits::eFragment},
     {ShaderEnding::COMP, "comp", vk::ShaderStageFlagBits::eCompute},
     {ShaderEnding::SPIR, "spir", vk::ShaderStageFlagBits::eAll},
     {ShaderEnding::UNKN, "", vk::ShaderStageFlagBits::eAll}}};


SAIGA_VULKAN_API EndingType getEnding(const std::string& file);
SAIGA_VULKAN_API std::string stripEnding(const std::string& file);

}  // namespace ShaderLoadHelper
}  // namespace Vulkan
}  // namespace Saiga
