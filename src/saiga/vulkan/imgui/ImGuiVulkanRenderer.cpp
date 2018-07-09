/*
* UI overlay class using ImGui
*
* Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "ImGuiVulkanRenderer.h"
#include "saiga/imgui/imgui.h"
#include "saiga/vulkan/Shader/ShaderPipeline.h"

#if defined(SAIGA_OPENGL_INCLUDED)
#error OpenGL was included somewhere.
#endif

namespace Saiga {
namespace Vulkan {

template<>
void VKVertexAttribBinder<ImDrawVert>::getVKAttribs(vk::VertexInputBindingDescription &vi_binding, std::vector<vk::VertexInputAttributeDescription> &attributeDescriptors)
{
    vi_binding.binding = 0;
    vi_binding.inputRate = vk::VertexInputRate::eVertex;
    vi_binding.stride = sizeof(ImDrawVert);

    attributeDescriptors.resize(3);

    attributeDescriptors[0].binding = 0;
    attributeDescriptors[0].location = 0;
    attributeDescriptors[0].format = vk::Format::eR32G32Sfloat;
    attributeDescriptors[0].offset = 0;

    attributeDescriptors[1].binding = 0;
    attributeDescriptors[1].location = 1;
    attributeDescriptors[1].format = vk::Format::eR32G32Sfloat;
    attributeDescriptors[1].offset = 1 * sizeof(vec2);

    attributeDescriptors[2].binding = 0;
    attributeDescriptors[2].location = 2;
    attributeDescriptors[2].format = vk::Format::eR8G8B8A8Unorm;
    attributeDescriptors[2].offset = 2 * sizeof(vec2);

}


ImGuiVulkanRenderer::~ImGuiVulkanRenderer()
{
    // Release all Vulkan resources required for rendering imGui
    vertexBuffer.destroy();
    indexBuffer.destroy();
    fontTexture.destroy();
//    vkDestroyImage(vulkanDevice->device, fontImage, nullptr);
//    vkDestroyImageView(vulkanDevice->device, fontView, nullptr);
//    vkFreeMemory(vulkanDevice->device, fontMemory, nullptr);
//    vkDestroySampler(vulkanDevice->device, sampler, nullptr);

    Pipeline::destroy();
}


void ImGuiVulkanRenderer::initResources(VulkanBase &_base, VkRenderPass renderPass)
{
    this->base = &_base;
    this->vulkanDevice = &_base;

    ImGuiIO& io = ImGui::GetIO();

    // Create font texture
    unsigned char* fontData;
    int texWidth, texHeight;
    io.Fonts->GetTexDataAsRGBA32(&fontData, &texWidth, &texHeight);

    ImageView<ucvec4> v(texHeight,texWidth,fontData);

    TemplatedImage<ucvec4> img(texHeight,texWidth);
    v.copyTo(img.getImageView());



    fontTexture.fromImage(_base,img);

    {
        device = this->vulkanDevice->device;

        uint32_t descriptorBindingPoint = 0;

        createDescriptorSetLayout({
                                      vk::DescriptorSetLayoutBinding{ descriptorBindingPoint,vk::DescriptorType::eCombinedImageSampler,1,vk::ShaderStageFlagBits::eFragment },
                                  });


        createPipelineLayout({
                                 vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex,0,sizeof(PushConstBlock))
                             });




        descriptorSet = createDescriptorSet();

        vk::DescriptorImageInfo fontDescriptor = fontTexture.getDescriptorInfo();

        device.updateDescriptorSets({
                                        vk::WriteDescriptorSet(descriptorSet,descriptorBindingPoint,0,1,vk::DescriptorType::eCombinedImageSampler,&fontDescriptor,nullptr,nullptr),
                                    },nullptr);

        shaderPipeline.load(
                    device,{
                        "vulkan/ui.vert",
                        "vulkan/ui.frag"
                    });

        PipelineInfo info;

        // Disable depth test and enable blending
        info.rasterizationState.cullMode = vk::CullModeFlagBits::eNone;
        info.depthStencilState.depthTestEnable = false;
        info.depthStencilState.depthWriteEnable = false;
        info.blendAttachmentState.blendEnable = true;

        info.addVertexInfo<ImDrawVert>();
        preparePipelines(info,vulkanDevice->pipelineCache,renderPass);
    }

//    VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, &vertexBuffer, maxVertexCount * sizeof(ImDrawVert)));
//    VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, &indexBuffer, maxIndexCount * sizeof(ImDrawIdx)));

    vertexBuffer.init(*base,std::vector<ImDrawVert>(maxVertexCount));
    indexBuffer.init (*base,std::vector<ImDrawIdx>(maxIndexCount));

//    vertexBuffer.map();
//    indexBuffer.map();
    cout << "Vulkan imgui created." << endl;

}

void ImGuiVulkanRenderer::updateBuffers()
{
    ImDrawData* imDrawData = ImGui::GetDrawData();

    // Note: Alignment is done inside buffer creation
    VkDeviceSize vertexBufferSize = imDrawData->TotalVtxCount * sizeof(ImDrawVert);
    VkDeviceSize indexBufferSize = imDrawData->TotalIdxCount * sizeof(ImDrawIdx);

    if(vertexBufferSize == 0 || indexBufferSize == 0)
        return;

    vertexCount = imDrawData->TotalVtxCount;
    indexCount = imDrawData->TotalIdxCount;

    // Upload data
//    ImDrawVert* vtxDst = (ImDrawVert*)vertexBuffer.mapped;
//    ImDrawIdx* idxDst = (ImDrawIdx*)indexBuffer.mapped;
    ImDrawVert* vtxDst = (ImDrawVert*)vertexBuffer.mapAll();
    ImDrawIdx* idxDst = (ImDrawIdx*)indexBuffer.mapAll();

    for (int n = 0; n < imDrawData->CmdListsCount; n++) {
        const ImDrawList* cmd_list = imDrawData->CmdLists[n];
        memcpy(vtxDst, cmd_list->VtxBuffer.Data, cmd_list->VtxBuffer.Size * sizeof(ImDrawVert));
        memcpy(idxDst, cmd_list->IdxBuffer.Data, cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx));
        vtxDst += cmd_list->VtxBuffer.Size;
        idxDst += cmd_list->IdxBuffer.Size;
    }

    // Flush to make writes visible to GPU
//    vertexBuffer.flush();
//    indexBuffer.flush();
    vertexBuffer.unmap();
    indexBuffer .unmap();
}

void ImGuiVulkanRenderer::render(vk::CommandBuffer commandBuffer)
{
    if(!vertexBuffer.buffer)
        return;

    ImGuiIO& io = ImGui::GetIO();

    //    vkCmdBindDescriptorSets(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
    //    vkCmdBindPipeline(commandBuffer, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

    vk::CommandBuffer cmd = commandBuffer;
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,pipelineLayout,0,descriptorSet,nullptr);
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics,pipeline);



    // Bind vertex and index buffer
    VkDeviceSize offsets[1] = { 0 };
//    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer.buffer, offsets);
//    vkCmdBindIndexBuffer(commandBuffer, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);
    vertexBuffer.bind(commandBuffer);
    indexBuffer.bind(commandBuffer);

    VkViewport viewport = vks::initializers::viewport(ImGui::GetIO().DisplaySize.x, ImGui::GetIO().DisplaySize.y, 0.0f, 1.0f);
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    // UI scale and translate via push constants
    pushConstBlock.scale = glm::vec2(2.0f / io.DisplaySize.x, 2.0f / io.DisplaySize.y);
    pushConstBlock.translate = glm::vec2(-1.0f);
    vkCmdPushConstants(commandBuffer, pipelineLayout, VK_SHADER_STAGE_VERTEX_BIT, 0, sizeof(PushConstBlock), &pushConstBlock);

    // Render commands
    ImDrawData* imDrawData = ImGui::GetDrawData();
    int32_t vertexOffset = 0;
    int32_t indexOffset = 0;
    for (int32_t i = 0; i < imDrawData->CmdListsCount; i++)
    {
        const ImDrawList* cmd_list = imDrawData->CmdLists[i];
        for (int32_t j = 0; j < cmd_list->CmdBuffer.Size; j++)
        {
            const ImDrawCmd* pcmd = &cmd_list->CmdBuffer[j];
            VkRect2D scissorRect;
            scissorRect.offset.x = std::max((int32_t)(pcmd->ClipRect.x), 0);
            scissorRect.offset.y = std::max((int32_t)(pcmd->ClipRect.y), 0);
            scissorRect.extent.width = (uint32_t)(pcmd->ClipRect.z - pcmd->ClipRect.x);
            scissorRect.extent.height = (uint32_t)(pcmd->ClipRect.w - pcmd->ClipRect.y);
            vkCmdSetScissor(commandBuffer, 0, 1, &scissorRect);
            vkCmdDrawIndexed(commandBuffer, pcmd->ElemCount, 1, indexOffset, vertexOffset, 0);
            indexOffset += pcmd->ElemCount;
        }
        vertexOffset += cmd_list->VtxBuffer.Size;
    }
}


void ImGuiVulkanRenderer::endFrame()
{
    ImGui::Render();

    updateBuffers();
}


}
}
