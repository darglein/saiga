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

    //    vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32_SFLOAT, offsetof(ImDrawVert, pos)),	// Location 0: Position
    //    vks::initializers::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32_SFLOAT, offsetof(ImDrawVert, uv)),	// Location 1: UV
    //    vks::initializers::vertexInputAttributeDescription(0, 2, VK_FORMAT_R8G8B8A8_UNORM, offsetof(ImDrawVert, col)),	// Location 0: Color


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
    vkDestroyImage(vulkanDevice->device, fontImage, nullptr);
    vkDestroyImageView(vulkanDevice->device, fontView, nullptr);
    vkFreeMemory(vulkanDevice->device, fontMemory, nullptr);
    vkDestroySampler(vulkanDevice->device, sampler, nullptr);

    Pipeline::destroy();
}


void ImGuiVulkanRenderer::initResources(VulkanBase &_base, VkRenderPass renderPass)
{
    this->vulkanDevice = &_base;

    ImGuiIO& io = ImGui::GetIO();

    // Create font texture
    unsigned char* fontData;
    int texWidth, texHeight;
    io.Fonts->GetTexDataAsRGBA32(&fontData, &texWidth, &texHeight);
    VkDeviceSize uploadSize = texWidth*texHeight * 4 * sizeof(char);

    // Create target image for copy
    VkImageCreateInfo imageInfo = vks::initializers::imageCreateInfo();
    imageInfo.imageType = VK_IMAGE_TYPE_2D;
    imageInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    imageInfo.extent.width = texWidth;
    imageInfo.extent.height = texHeight;
    imageInfo.extent.depth = 1;
    imageInfo.mipLevels = 1;
    imageInfo.arrayLayers = 1;
    imageInfo.samples = VK_SAMPLE_COUNT_1_BIT;
    imageInfo.tiling = VK_IMAGE_TILING_OPTIMAL;
    imageInfo.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    imageInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    imageInfo.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    VK_CHECK_RESULT(vkCreateImage(vulkanDevice->device, &imageInfo, nullptr, &fontImage));
    VkMemoryRequirements memReqs;
    vkGetImageMemoryRequirements(vulkanDevice->device, fontImage, &memReqs);
    VkMemoryAllocateInfo memAllocInfo = vks::initializers::memoryAllocateInfo();
    memAllocInfo.allocationSize = memReqs.size;
    memAllocInfo.memoryTypeIndex = vulkanDevice->getMemoryType(memReqs.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK_RESULT(vkAllocateMemory(vulkanDevice->device, &memAllocInfo, nullptr, &fontMemory));
    VK_CHECK_RESULT(vkBindImageMemory(vulkanDevice->device, fontImage, fontMemory, 0));

    // Image view
    VkImageViewCreateInfo viewInfo = vks::initializers::imageViewCreateInfo();
    viewInfo.image = fontImage;
    viewInfo.viewType = VK_IMAGE_VIEW_TYPE_2D;
    viewInfo.format = VK_FORMAT_R8G8B8A8_UNORM;
    viewInfo.subresourceRange.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    viewInfo.subresourceRange.levelCount = 1;
    viewInfo.subresourceRange.layerCount = 1;
    VK_CHECK_RESULT(vkCreateImageView(vulkanDevice->device, &viewInfo, nullptr, &fontView));

    // Staging buffers for font data upload
    vks::Buffer stagingBuffer;

    VK_CHECK_RESULT(vulkanDevice->createBuffer(
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        &stagingBuffer,
                        uploadSize));

    stagingBuffer.map();
    memcpy(stagingBuffer.mapped, fontData, uploadSize);
    stagingBuffer.unmap();

    // Copy buffer data to font image
//    vk::CommandBuffer cmd = vulkanDevice->createCommandBuffer(VK_COMMAND_BUFFER_LEVEL_PRIMARY, false);
    VkCommandBuffer cmd = vulkanDevice->createAndBeginTransferCommand();


//    cmd.begin(vk::CommandBufferBeginInfo());

    // Prepare for transfer
    vks::tools::setImageLayout(
                cmd,
                fontImage,
                VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_UNDEFINED,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_HOST_BIT,
                VK_PIPELINE_STAGE_TRANSFER_BIT);

    // Copy
    VkBufferImageCopy bufferCopyRegion = {};
    bufferCopyRegion.imageSubresource.aspectMask = VK_IMAGE_ASPECT_COLOR_BIT;
    bufferCopyRegion.imageSubresource.layerCount = 1;
    bufferCopyRegion.imageExtent.width = texWidth;
    bufferCopyRegion.imageExtent.height = texHeight;
    bufferCopyRegion.imageExtent.depth = 1;

    vkCmdCopyBufferToImage(
                cmd,
                stagingBuffer.buffer,
                fontImage,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                1,
                &bufferCopyRegion
                );

    // Prepare for shader read
    vks::tools::setImageLayout(
                cmd,
                fontImage,
                VK_IMAGE_ASPECT_COLOR_BIT,
                VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);

//    cmd.end();
//        vulkanDevice->transferAndWait(cmd, true);
    vulkanDevice->endTransferWait(cmd);
//    copyQueue.submitAndWait(cmd);

    stagingBuffer.destroy();

    // Font texture Sampler
    VkSamplerCreateInfo samplerInfo = vks::initializers::samplerCreateInfo();
    samplerInfo.magFilter = VK_FILTER_LINEAR;
    samplerInfo.minFilter = VK_FILTER_LINEAR;
    samplerInfo.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    samplerInfo.addressModeU = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeV = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    samplerInfo.borderColor = VK_BORDER_COLOR_FLOAT_OPAQUE_WHITE;
    VK_CHECK_RESULT(vkCreateSampler(vulkanDevice->device, &samplerInfo, nullptr, &sampler));

    {
        device = this->vulkanDevice->device;

        uint32_t descriptorBindingPoint = 0;

        createDescriptorSetLayout({
                                      vk::DescriptorSetLayoutBinding{ descriptorBindingPoint,vk::DescriptorType::eCombinedImageSampler,1,vk::ShaderStageFlagBits::eFragment },
                                  });


        createPipelineLayout({
                                 vk::PushConstantRange(vk::ShaderStageFlagBits::eVertex,0,sizeof(PushConstBlock))
                             });


        createDescriptorPool(
                    1,{
                        vk::DescriptorPoolSize{vk::DescriptorType::eCombinedImageSampler, 1}
                    });



        descriptorSet = device.allocateDescriptorSets(
                    vk::DescriptorSetAllocateInfo(descriptorPool,descriptorSetLayout.size(),descriptorSetLayout.data())
                    );


        vk::DescriptorImageInfo fontDescriptor = vks::initializers::descriptorImageInfo(
                    sampler,
                    fontView,
                    VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL
                    );

        //        vk::DescriptorBufferInfo descriptorInfo =fontDescriptor;
        device.updateDescriptorSets({
                                        vk::WriteDescriptorSet(descriptorSet[0],descriptorBindingPoint,0,1,vk::DescriptorType::eCombinedImageSampler,&fontDescriptor,nullptr,nullptr),
                                    },nullptr);

        //        vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 0, &fontDescriptor)

        // Load all shaders.
        // Note: The shader type is deduced from the ending.
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

    VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, &vertexBuffer, maxVertexCount * sizeof(ImDrawVert)));
    VK_CHECK_RESULT(vulkanDevice->createBuffer(VK_BUFFER_USAGE_INDEX_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT, &indexBuffer, maxIndexCount * sizeof(ImDrawIdx)));

    vertexBuffer.map();
    indexBuffer.map();
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
    ImDrawVert* vtxDst = (ImDrawVert*)vertexBuffer.mapped;
    ImDrawIdx* idxDst = (ImDrawIdx*)indexBuffer.mapped;

    for (int n = 0; n < imDrawData->CmdListsCount; n++) {
        const ImDrawList* cmd_list = imDrawData->CmdLists[n];
        memcpy(vtxDst, cmd_list->VtxBuffer.Data, cmd_list->VtxBuffer.Size * sizeof(ImDrawVert));
        memcpy(idxDst, cmd_list->IdxBuffer.Data, cmd_list->IdxBuffer.Size * sizeof(ImDrawIdx));
        vtxDst += cmd_list->VtxBuffer.Size;
        idxDst += cmd_list->IdxBuffer.Size;
    }

    // Flush to make writes visible to GPU
    vertexBuffer.flush();
    indexBuffer.flush();
}

void ImGuiVulkanRenderer::render(VkCommandBuffer commandBuffer)
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
    vkCmdBindVertexBuffers(commandBuffer, 0, 1, &vertexBuffer.buffer, offsets);
    vkCmdBindIndexBuffer(commandBuffer, indexBuffer.buffer, 0, VK_INDEX_TYPE_UINT16);

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
