/*
 * UI overlay class using ImGui
 *
 * Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include "ImGuiVulkanRenderer.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/math/imath.h"
#include "saiga/core/util/fileChecker.h"

#if defined(SAIGA_OPENGL_INCLUDED)
#    error OpenGL was included somewhere.
#endif

namespace Saiga
{
namespace Vulkan
{
template <>
void VKVertexAttribBinder<ImDrawVert>::getVKAttribs(
    vk::VertexInputBindingDescription& vi_binding,
    std::vector<vk::VertexInputAttributeDescription>& attributeDescriptors)
{
    vi_binding.binding   = 0;
    vi_binding.inputRate = vk::VertexInputRate::eVertex;
    vi_binding.stride    = sizeof(ImDrawVert);

    attributeDescriptors.resize(3);

    attributeDescriptors[0].binding  = 0;
    attributeDescriptors[0].location = 0;
    attributeDescriptors[0].format   = vk::Format::eR32G32Sfloat;
    attributeDescriptors[0].offset   = 0;

    attributeDescriptors[1].binding  = 0;
    attributeDescriptors[1].location = 1;
    attributeDescriptors[1].format   = vk::Format::eR32G32Sfloat;
    attributeDescriptors[1].offset   = 1 * sizeof(vec2);

    attributeDescriptors[2].binding  = 0;
    attributeDescriptors[2].location = 2;
    attributeDescriptors[2].format   = vk::Format::eR8G8B8A8Unorm;
    attributeDescriptors[2].offset   = 2 * sizeof(vec2);
}


ImGuiVulkanRenderer::ImGuiVulkanRenderer(size_t _frameCount, const ImGuiParameters& params)
    : ImGuiRenderer(params), frameCount(_frameCount), frameData()
{
}

ImGuiVulkanRenderer::~ImGuiVulkanRenderer()
{
    for (auto& data : frameData)
    {
        data.destroy(*base);
    }
    fontTexture.destroy();
    Pipeline::destroy();
}


void ImGuiVulkanRenderer::initResources(VulkanBase& _base, VkRenderPass renderPass)
{
    this->base         = &_base;
    this->vulkanDevice = &_base;


    PipelineBase::init(_base, 1);

    ImGuiIO& io = ImGui::GetIO();



    // Create font texture
    unsigned char* fontData;
    int texWidth, texHeight;
    io.Fonts->GetTexDataAsRGBA32(&fontData, &texWidth, &texHeight);



    ImageView<ucvec4> v(texHeight, texWidth, fontData);

    TemplatedImage<ucvec4> img(texHeight, texWidth);
    v.copyTo(img.getImageView());



    fontTexture.fromImage(_base, img, vk::ImageUsageFlagBits::eSampled, false);

    {
        device = this->vulkanDevice->device;

        uint32_t textureBindingPoint = 0;

        addDescriptorSetLayout({{0,
                                 {textureBindingPoint, vk::DescriptorType::eCombinedImageSampler, 1,
                                  vk::ShaderStageFlagBits::eFragment}}});
        addPushConstantRange({vk::ShaderStageFlagBits::eVertex, 0, sizeof(PushConstBlock)});

        descriptorSet = createDescriptorSet();

        vk::DescriptorImageInfo fontDescriptor = fontTexture.getDescriptorInfo();

        device.updateDescriptorSets(
            {
                vk::WriteDescriptorSet(descriptorSet, textureBindingPoint, 0, 1,
                                       vk::DescriptorType::eCombinedImageSampler, &fontDescriptor, nullptr, nullptr),
            },
            nullptr);

        shaderPipeline.load(device, {"vulkan/ui.vert", "vulkan/ui.frag"});

        PipelineInfo info;
        // Disable depth test and enable blending
        info.rasterizationState.cullMode        = vk::CullModeFlagBits::eNone;
        info.depthStencilState.depthTestEnable  = VK_FALSE;
        info.depthStencilState.depthWriteEnable = VK_FALSE;
        info.blendAttachmentState.blendEnable   = VK_TRUE;
        info.addVertexInfo<ImDrawVert>();
        create(renderPass, info);
    }

    frameData.reserve(frameCount);
    for (auto i = 0UL; i < frameCount; ++i)
    {
        frameData.emplace_back(*base, initialMaxVertexCount, initialMaxIndexCount);
    }

    std::cout << "Vulkan imgui created." << std::endl;
}

void ImGuiVulkanRenderer::updateBuffers(vk::CommandBuffer cmd, size_t index)
{
    ImDrawData* imDrawData = ImGui::GetDrawData();
    SAIGA_ASSERT(imDrawData);

    vertexCount = imDrawData->TotalVtxCount;
    indexCount  = imDrawData->TotalIdxCount;

    if (vertexCount == 0 || indexCount == 0)
    {
        return;
    }


    auto& currentFrameData = frameData[index];

    currentFrameData.resizeIfNecessary(*base, vertexCount, indexCount);

    ImDrawVert* vtxDst = currentFrameData.vertexData;
    ImDrawIdx* idxDst  = currentFrameData.indexData;

    for (int n = 0; n < imDrawData->CmdListsCount; n++)
    {
        const ImDrawList* cmd_list = imDrawData->CmdLists[n];
        memcpy(vtxDst, cmd_list->VtxBuffer.Data, cmd_list->VtxBuffer.Size * sizeof(ImDrawVert));
        memcpy(idxDst, cmd_list->IdxBuffer.Data, cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx));
        vtxDst += cmd_list->VtxBuffer.Size;
        idxDst += cmd_list->IdxBuffer.Size;
    }
    // Flush to make writes visible to GPU
    // currentFrameData.vertexBuffer.flush(*base);
    // currentFrameData.indexBuffer.flush(*base);
}

void ImGuiVulkanRenderer::render(vk::CommandBuffer commandBuffer, size_t frameIndex)
{
    if (vertexCount == 0 || indexCount == 0) return;

    ImGuiIO& io = ImGui::GetIO();

    vk::CommandBuffer cmd = commandBuffer;

    if (!Pipeline::bind(commandBuffer)) return;
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, descriptorSet, nullptr);
    //    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);

    //    vertexBuffer.bind(commandBuffer);
    //    indexBuffer.bind(commandBuffer);

    VkViewport viewport =
        vks::initializers::viewport(ImGui::GetIO().DisplaySize.x, ImGui::GetIO().DisplaySize.y, 0.0f, 1.0f);
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    // UI scale and translate via push constants
    pushConstBlock.scale     = vec2(2.0f / io.DisplaySize.x, 2.0f / io.DisplaySize.y);
    pushConstBlock.translate = make_vec2(-1.0f);
    pushConstant(cmd, vk::ShaderStageFlagBits::eVertex, sizeof(PushConstBlock), &pushConstBlock);

    // Render commands
    ImDrawData* imDrawData = ImGui::GetDrawData();
    int32_t vertexOffset   = 0;
    uint32_t indexOffset   = 0;

    if (imDrawData->CmdListsCount > 0)
    {
        auto& currentFrameData = frameData[frameIndex];
        currentFrameData.vertexBuffer.bind(commandBuffer);
        currentFrameData.indexBuffer.bind(commandBuffer);
        for (int32_t i = 0; i < imDrawData->CmdListsCount; i++)
        {
            const ImDrawList* cmd_list = imDrawData->CmdLists[i];
            for (int32_t j = 0; j < cmd_list->CmdBuffer.Size; j++)
            {
                const ImDrawCmd* pcmd = &cmd_list->CmdBuffer[j];
                VkRect2D scissorRect;
                scissorRect.offset.x      = std::max((int32_t)(pcmd->ClipRect.x), 0);
                scissorRect.offset.y      = std::max((int32_t)(pcmd->ClipRect.y), 0);
                scissorRect.extent.width  = (uint32_t)(pcmd->ClipRect.z - std::max((int32_t)(pcmd->ClipRect.x), 0));
                scissorRect.extent.height = (uint32_t)(pcmd->ClipRect.w - std::max((int32_t)(pcmd->ClipRect.y), 0));
                vkCmdSetScissor(commandBuffer, 0, 1, &scissorRect);
                vkCmdDrawIndexed(commandBuffer, pcmd->ElemCount, 1, indexOffset, vertexOffset, 0);
                indexOffset += pcmd->ElemCount;
            }
            vertexOffset += cmd_list->VtxBuffer.Size;
        }
    }
}


void ImGuiVulkanRenderer::endFrame()
{
    ImGui::Render();
}



ImGuiVulkanRenderer::FrameData::FrameData(VulkanBase& base, const uint32_t maxVertexCount, const uint32_t maxIndexCount)
{
    indexBuffer.init(base, maxIndexCount,
                     vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    vertexBuffer.init(base, maxVertexCount,
                      vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    SAIGA_ASSERT(vertexBuffer.isMapped() && indexBuffer.isMapped(), "ImGui buffers must be mapped");

    vertexData = (ImDrawVert*)vertexBuffer.getMappedPointer();
    indexData  = (ImDrawIdx*)indexBuffer.getMappedPointer();
}

void ImGuiVulkanRenderer::FrameData::destroy(VulkanBase& base)
{
    vertexBuffer.destroy();
    indexBuffer.destroy();
}
}  // namespace Vulkan
}  // namespace Saiga
