/**
 * Copyright (c) 2021 Darius Rückert
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

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/imgui/imgui_renderer.h"
#include "saiga/core/math/math.h"
#include "saiga/vulkan/Base.h"
#include "saiga/vulkan/buffer/IndexBuffer.h"
#include "saiga/vulkan/buffer/VertexBuffer.h"
#include "saiga/vulkan/pipeline/Pipeline.h"
#include "saiga/vulkan/texture/Texture.h"


namespace Saiga
{
namespace Vulkan
{
class SAIGA_VULKAN_API ImGuiVulkanRenderer : public Pipeline, public ImGuiRenderer
{
   private:
    struct FrameData
    {
        VertexBuffer<ImDrawVert> vertexBuffer;
        IndexBuffer<ImDrawIdx> indexBuffer;
        ImDrawVert* vertexData;
        ImDrawIdx* indexData;

        FrameData(VulkanBase& base, uint32_t maxVertexCount, uint32_t maxIndexCount);

        FrameData(const FrameData& other) = delete;
        FrameData(FrameData&& other)      = default;

        FrameData& operator=(const FrameData& other) = delete;
        FrameData& operator=(FrameData&& other) = default;

        void destroy(VulkanBase& base);

        void resizeIfNecessary(VulkanBase& base, uint32_t vertexCount, uint32_t indexCount)
        {
            if (vertexCount > vertexBuffer.vertexCount)
            {
                uint32_t currentCount = vertexBuffer.vertexCount;
                while (currentCount < vertexCount)
                {
                    currentCount = currentCount << 1;
                }
                vertexBuffer.destroy();
                vertexBuffer.init(base, currentCount,
                                  vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

                vertexData = static_cast<ImDrawVert*>(vertexBuffer.getMappedPointer());
                VLOG(3) << "Resizing imgui vertex buffer: " << currentCount << " vertices";
            }
            if (indexCount > indexBuffer.indexCount)
            {
                uint32_t currentCount = indexBuffer.indexCount;
                while (currentCount < indexCount)
                {
                    currentCount = currentCount << 1;
                }
                indexBuffer.destroy();
                indexBuffer.init(base, currentCount,
                                 vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

                indexData = static_cast<ImDrawIdx*>(indexBuffer.getMappedPointer());
                VLOG(3) << "Resizing imgui index buffer: " << currentCount << " vertices";
            }
        }
    };

    size_t frameCount;
    std::vector<FrameData> frameData;

   public:
    ImGuiVulkanRenderer(size_t _frameCount, const ImGuiParameters& params);
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
    int32_t indexCount  = 0;

    uint32_t initialMaxVertexCount = 64 * 1024;
    uint32_t initialMaxIndexCount  = 64 * 1024;


    Texture2D fontTexture;

    StaticDescriptorSet descriptorSet;
    Saiga::Vulkan::VulkanBase* vulkanDevice;

    double g_Time = 0.0f;
    bool g_MousePressed[3];
    float g_MouseWheel = 0.0f;
};

}  // namespace Vulkan
}  // namespace Saiga
