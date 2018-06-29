/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "Buffer.h"
#include "saiga/vulkan/vulkanBase.h"

namespace Saiga {
namespace Vulkan {


class SAIGA_GLOBAL UniformBuffer : public Buffer
{
public:


    glm::mat4 Projection;
    glm::mat4 View;
    glm::mat4 Model;
    glm::mat4 Clip;
    glm::mat4 MVP;

    void init(VulkanBase& base);
};

}
}
