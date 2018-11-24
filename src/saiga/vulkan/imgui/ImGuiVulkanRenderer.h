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

#include "saiga/util/math.h"

#include "saiga/vulkan/Base.h"
#include "saiga/vulkan/pipeline/Pipeline.h"
#include "saiga/vulkan/texture/Texture.h"
#include "saiga/vulkan/buffer/VertexBuffer.h"
#include "saiga/vulkan/buffer/IndexBuffer.h"

#include "saiga/imgui/imgui.h"


namespace Saiga {
namespace Vulkan {

class SAIGA_GLOBAL ImGuiVulkanRenderer : public Pipeline
{
private:
    struct FrameData{
        VertexBuffer<ImDrawVert> vertexBuffer;
        IndexBuffer<ImDrawIdx> indexBuffer;
        ImDrawVert* vertexData;
        ImDrawIdx* indexData;

        FrameData(VulkanBase& base, const uint32_t maxVertexCount, const uint32_t maxIndexCount) {
            vertexBuffer.init(base,maxVertexCount, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
            indexBuffer.init(base,maxIndexCount,  vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

            SAIGA_ASSERT(vertexBuffer.isMapped() && indexBuffer.isMapped(), "ImGui buffers must be mapped");

            vertexData = (ImDrawVert *) vertexBuffer.getMappedPointer();
            indexData = (ImDrawIdx *) indexBuffer.getMappedPointer();
        }

        void destroy(VulkanBase& base) {
            vertexBuffer.destroy();
            indexBuffer.destroy();
        }
    };

    size_t frameCount;
    std::vector<FrameData> frameData;
public:
    ImGuiVulkanRenderer(size_t _frameCount) : frameCount(_frameCount), frameData(){
        ImGui::CreateContext();
    }

    virtual ~ImGuiVulkanRenderer();

    void initResources(Saiga::Vulkan::VulkanBase& vulkanDevice, VkRenderPass renderPass);
    virtual void beginFrame() = 0;
    void endFrame();

    void updateBuffers(vk::CommandBuffer cmd, size_t frameIndex);
    void render(vk::CommandBuffer commandBuffer, size_t frameIndex);
protected:

    struct PushConstBlock
    {
        vec2 scale;
        vec2 translate;
    } pushConstBlock;

    int32_t vertexCount = 0;
    int32_t indexCount = 0;

    int32_t maxVertexCount = 50000;
    int32_t maxIndexCount = 50000;


    Texture2D fontTexture;

    vk::DescriptorSet       descriptorSet;
    Saiga::Vulkan::VulkanBase *vulkanDevice;

    double       g_Time = 0.0f;
    bool         g_MousePressed[3];
    float        g_MouseWheel = 0.0f;
};

}
}
