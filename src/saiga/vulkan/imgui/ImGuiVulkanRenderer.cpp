/*
* UI overlay class using ImGui
*
* Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "ImGuiVulkanRenderer.h"
#include "saiga/imgui/imgui.h"

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
    vertexBuffer.destroy();
    indexBuffer.destroy();
    fontTexture.destroy(*base);
    Pipeline::destroy();
}


void ImGuiVulkanRenderer::initResources(VulkanBase &_base, VkRenderPass renderPass)
{
    this->base = &_base;
    this->vulkanDevice = &_base;

    PipelineBase::init(_base,1);

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

        uint32_t textureBindingPoint = 0;

        addDescriptorSetLayout({{ textureBindingPoint,vk::DescriptorType::eCombinedImageSampler,1,vk::ShaderStageFlagBits::eFragment }});
        addPushConstantRange( {vk::ShaderStageFlagBits::eVertex,0,sizeof(PushConstBlock)} );

        descriptorSet = createDescriptorSet();

        vk::DescriptorImageInfo fontDescriptor = fontTexture.getDescriptorInfo();

        device.updateDescriptorSets({
                                        vk::WriteDescriptorSet(descriptorSet,textureBindingPoint,0,1,vk::DescriptorType::eCombinedImageSampler,&fontDescriptor,nullptr,nullptr),
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
        create(renderPass,info);
    }

    /*
     *  We use host visible memory here because each vertex and index is read only once by the GPU.
     *  A slightly better performance can be obtained by creating a second device only buffer and copying the
     * data asynchron in a transfer queue. Then the data might be already present when render is called.
     */
    vertexBuffer.init(*base,maxVertexCount, vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    indexBuffer.init (*base,maxIndexCount,  vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);

    vertexData = (ImDrawVert*)vertexBuffer.m_memoryLocation.map(base->device);
    indexData = (ImDrawIdx*)indexBuffer.m_memoryLocation.map(base->device);

    cout << "Vulkan imgui created." << endl;
}

void ImGuiVulkanRenderer::updateBuffers(vk::CommandBuffer cmd)
{
    ImDrawData* imDrawData = ImGui::GetDrawData();
    SAIGA_ASSERT(imDrawData);

    vertexCount = imDrawData->TotalVtxCount;
    indexCount = imDrawData->TotalIdxCount;

    if(vertexCount == 0 || indexCount == 0)
        return;

    ImDrawVert* vtxDst = vertexData;
    ImDrawIdx* idxDst = indexData  ;

    for (int n = 0; n < imDrawData->CmdListsCount; n++)
    {
        const ImDrawList* cmd_list = imDrawData->CmdLists[n];
        memcpy(vtxDst, cmd_list->VtxBuffer.Data, cmd_list->VtxBuffer.Size * sizeof(ImDrawVert));
        memcpy(idxDst, cmd_list->IdxBuffer.Data, cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx));
        vtxDst += cmd_list->VtxBuffer.Size;
        idxDst += cmd_list->IdxBuffer.Size;
    }
}

void ImGuiVulkanRenderer::render(vk::CommandBuffer commandBuffer)
{
//    if(!vertexBuffer.buffer)
//        return;

    if(vertexCount == 0 || indexCount == 0)
        return;

    ImGuiIO& io = ImGui::GetIO();

    vk::CommandBuffer cmd = commandBuffer;
    cmd.bindDescriptorSets(vk::PipelineBindPoint::eGraphics,pipelineLayout,0,descriptorSet,nullptr);
    cmd.bindPipeline(vk::PipelineBindPoint::eGraphics,pipeline);

    vertexBuffer.bind(commandBuffer);
    indexBuffer.bind(commandBuffer);

    VkViewport viewport = vks::initializers::viewport(ImGui::GetIO().DisplaySize.x, ImGui::GetIO().DisplaySize.y, 0.0f, 1.0f);
    vkCmdSetViewport(commandBuffer, 0, 1, &viewport);

    // UI scale and translate via push constants
    pushConstBlock.scale = glm::vec2(2.0f / io.DisplaySize.x, 2.0f / io.DisplaySize.y);
    pushConstBlock.translate = glm::vec2(-1.0f);
    pushConstant(cmd,vk::ShaderStageFlagBits::eVertex,sizeof(PushConstBlock),&pushConstBlock);

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
}


}
}
