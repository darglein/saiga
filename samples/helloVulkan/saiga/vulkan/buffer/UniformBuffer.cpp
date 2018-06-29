/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "UniformBuffer.h"
#include "saiga/vulkan/vulkanHelper.h"

namespace Saiga {
namespace Vulkan {

void UniformBuffer::init(VulkanBase &base)
{
    vk::Result res;
    Projection = glm::perspective(glm::radians(45.0f), 1.0f, 0.1f, 100.0f);
//    View = glm::lookAt(glm::vec3(0,0,-50),  // Camera is at (-5,3,-10), in World Space
//                            glm::vec3(0, 0, 0),     // and looks at the origin
//                            glm::vec3(0, 1, 0)     // Head is up (set to 0,-1,0 to look upside-down)
//                            );

    View = glm::translate(glm::vec3(0.0f, 0.0f, -5.5));
    Model = glm::mat4(1.0f);

    // Vulkan clip space has inverted Y and half Z.
    // clang-format off
    Clip = glm::mat4(1.0f, 0.0f, 0.0f, 0.0f,
                          0.0f,-1.0f, 0.0f, 0.0f,
                          0.0f, 0.0f, 0.5f, 0.0f,
                          0.0f, 0.0f, 0.5f, 1.0f);
    // clang-format on
    MVP =  Projection * View * Model;


    createBuffer(base,sizeof(glm::mat4),vk::BufferUsageFlagBits::eUniformBuffer);
    allocateMemory(base);
    upload(base,0,sizeof(glm::mat4),&MVP[0][0]);


    base.device.bindBufferMemory(buffer,memory,0);

    info.buffer = buffer;
    info.offset = 0;
    info.range = sizeof(MVP);

}




}
}
