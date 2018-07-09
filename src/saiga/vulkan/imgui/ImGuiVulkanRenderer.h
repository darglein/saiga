/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
/*
* UI overlay class using ImGui
*
* Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#pragma once

#include "saiga/util/glm.h"

#include "saiga/vulkan/VulkanBuffer.hpp"
#include "saiga/vulkan/Base.h"
#include <saiga/sdl/sdl_eventhandler.h>
#include "saiga/vulkan/Shader/ShaderPipeline.h"
#include "saiga/vulkan/pipeline/Pipeline.h"

#include "saiga/vulkan/Queue.h"


namespace Saiga {
namespace Vulkan {

class SAIGA_GLOBAL ImGuiVulkanRenderer : public Pipeline
{
public:
    ~ImGuiVulkanRenderer();


    // Initialize all Vulkan resources used by the ui
    void initResources(Saiga::Vulkan::VulkanBase& vulkanDevice, VkRenderPass renderPass);


    // Draw current imGui frame into a command buffer
    void render(VkCommandBuffer commandBuffer);

    virtual void beginFrame() = 0;
    void endFrame();

protected:

    // UI params are set via push constants
    struct PushConstBlock {
        glm::vec2 scale;
        glm::vec2 translate;
    } pushConstBlock;

    SDL_Window* window;
    // Vulkan resources for rendering the UI
    VkSampler sampler;
    vks::Buffer vertexBuffer;
    vks::Buffer indexBuffer;
    int32_t vertexCount = 0;
    int32_t indexCount = 0;

    int32_t maxVertexCount = 10000;
    int32_t maxIndexCount = 10000;


    VkDeviceMemory fontMemory = VK_NULL_HANDLE;
    VkImage fontImage = VK_NULL_HANDLE;
    VkImageView fontView = VK_NULL_HANDLE;

    vk::DescriptorSet       descriptorSet;
    Saiga::Vulkan::VulkanBase *vulkanDevice;

//    Saiga::Vulkan::ShaderPipeline shaderPipeline;

    double       g_Time = 0.0f;
    bool         g_MousePressed[3];
    float        g_MouseWheel = 0.0f;
    // Update vertex and index buffer containing the imGui elements when required
    void updateBuffers();
};

}
}
