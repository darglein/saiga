/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Shader.h"
#include "saiga/vulkan/vulkanHelper.h"

#include "GLSL.h"

namespace Saiga {
namespace Vulkan {


void VShader::destroy()
{

}

std::vector<vk::PipelineShaderStageCreateInfo> VShader::createPipelineInfo()
{
   std::vector<vk::PipelineShaderStageCreateInfo> info(modules.size());

   for(unsigned int i = 0;  i < modules.size(); ++i)
   {
       info[i].pSpecializationInfo = NULL;
       info[i].stage = modules[i].stage;
       info[i].pName = "main";
   }
   return info;
}




}
}
