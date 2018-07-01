/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once


#include "saiga/vulkan/vulkan.h"
#include "saiga/vulkan/Shader/ShaderModule.h"
#include "saiga/vulkan/Shader/Shader.h"

namespace Saiga {
namespace Vulkan {
namespace ShaderLoadHelper{

enum class ShaderEnding
{
    VERT,
    TESC,
    TESE,
    GEOM,
    FRAG,
    COMP,
    SPIR,
    UNKN // unknown ending
};

using EndingType = std::tuple<ShaderEnding,std::string,vk::ShaderStageFlagBits>;

const std::array<EndingType,8> fileEndings =
{{
     { ShaderEnding::VERT,  "vert", vk::ShaderStageFlagBits::eVertex                  },
     { ShaderEnding::TESC,  "tesc", vk::ShaderStageFlagBits::eTessellationControl     },
     { ShaderEnding::TESE,  "tese", vk::ShaderStageFlagBits::eTessellationEvaluation  },
     { ShaderEnding::GEOM,  "geom", vk::ShaderStageFlagBits::eGeometry                },
     { ShaderEnding::FRAG,  "frag", vk::ShaderStageFlagBits::eFragment                },
     { ShaderEnding::COMP,  "comp", vk::ShaderStageFlagBits::eCompute                 },
     { ShaderEnding::SPIR,  "spir", vk::ShaderStageFlagBits::eAll                     },
     { ShaderEnding::UNKN,  "", vk::ShaderStageFlagBits::eAll                         }
 }};


EndingType getEnding(const std::string& file);
std::string stripEnding(const std::string& file);

}
}
}
