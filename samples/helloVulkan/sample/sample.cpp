/*
* Vulkan Example - imGui (https://github.com/ocornut/imgui)
*
* Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "sample.h"

VulkanExample::VulkanExample() : VulkanExampleBase(ENABLE_VALIDATION)
{
    title = "Vulkan Example - ImGui";
    camera.type = Camera::CameraType::lookat;
    camera.setPosition(glm::vec3(0.0f, 1.4f, -4.8f));
    camera.setRotation(glm::vec3(4.5f, -380.0f, 0.0f));
    camera.setPerspective(45.0f, (float)width / (float)height, 0.1f, 256.0f);
}

VulkanExample::~VulkanExample()
{
    vkDestroyPipeline(device, pipeline, nullptr);
    vkDestroyPipelineLayout(device, pipelineLayout, nullptr);
    vkDestroyDescriptorSetLayout(device, descriptorSetLayout, nullptr);

    models.models.destroy();
    models.background.destroy();
    models.logos.destroy();

    uniformBufferVS.destroy();

}

void VulkanExample::buildCommandBuffers()
{
    VkCommandBufferBeginInfo cmdBufInfo = vks::initializers::commandBufferBeginInfo();

    VkClearValue clearValues[2];
    clearValues[0].color = { { 0.2f, 0.2f, 0.2f, 1.0f} };
    clearValues[1].depthStencil = { 1.0f, 0 };

    VkRenderPassBeginInfo renderPassBeginInfo = vks::initializers::renderPassBeginInfo();
    renderPassBeginInfo.renderPass = renderPass;
    renderPassBeginInfo.renderArea.offset.x = 0;
    renderPassBeginInfo.renderArea.offset.y = 0;
    renderPassBeginInfo.renderArea.extent.width = width;
    renderPassBeginInfo.renderArea.extent.height = height;
    renderPassBeginInfo.clearValueCount = 2;
    renderPassBeginInfo.pClearValues = clearValues;

    imGui->newFrame(this, (frameCounter == 0));

    imGui->updateBuffers();

    for (int32_t i = 0; i < drawCmdBuffers.size(); ++i)
    {
        // Set target frame buffer
        renderPassBeginInfo.framebuffer = frameBuffers[i];

        VK_CHECK_RESULT(vkBeginCommandBuffer(drawCmdBuffers[i], &cmdBufInfo));

        vkCmdBeginRenderPass(drawCmdBuffers[i], &renderPassBeginInfo, VK_SUBPASS_CONTENTS_INLINE);

        VkViewport viewport = vks::initializers::viewport((float)width, (float)height, 0.0f, 1.0f);
        vkCmdSetViewport(drawCmdBuffers[i], 0, 1, &viewport);

        VkRect2D scissor = vks::initializers::rect2D(width, height, 0, 0);
        vkCmdSetScissor(drawCmdBuffers[i], 0, 1, &scissor);

        // Render scene
        vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
        vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

        VkDeviceSize offsets[1] = { 0 };
        if (uiSettings.displayBackground) {
            vkCmdBindVertexBuffers(drawCmdBuffers[i], 0, 1, &models.background.vertices.buffer, offsets);
            vkCmdBindIndexBuffer(drawCmdBuffers[i], models.background.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(drawCmdBuffers[i], models.background.indexCount, 1, 0, 0, 0);
        }

        if (uiSettings.displayModels) {
            vkCmdBindVertexBuffers(drawCmdBuffers[i], 0, 1, &models.models.vertices.buffer, offsets);
            vkCmdBindIndexBuffer(drawCmdBuffers[i], models.models.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(drawCmdBuffers[i], models.models.indexCount, 1, 0, 0, 0);
        }

        if (uiSettings.displayLogos) {
            vkCmdBindVertexBuffers(drawCmdBuffers[i], 0, 1, &models.logos.vertices.buffer, offsets);
            vkCmdBindIndexBuffer(drawCmdBuffers[i], models.logos.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
            vkCmdDrawIndexed(drawCmdBuffers[i], models.logos.indexCount, 1, 0, 0, 0);
        }

        // Render imGui
        imGui->drawFrame(drawCmdBuffers[i]);

        vkCmdEndRenderPass(drawCmdBuffers[i]);

        VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
    }
}

void VulkanExample::setupLayoutsAndDescriptors()
{
    // descriptor pool
    std::vector<VkDescriptorPoolSize> poolSizes = {
        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 2),
        vks::initializers::descriptorPoolSize(VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1)
    };
    VkDescriptorPoolCreateInfo descriptorPoolInfo = vks::initializers::descriptorPoolCreateInfo(poolSizes, 2);
    VK_CHECK_RESULT(vkCreateDescriptorPool(device, &descriptorPoolInfo, nullptr, &descriptorPool));

    // Set layout
    std::vector<VkDescriptorSetLayoutBinding> setLayoutBindings = {
        vks::initializers::descriptorSetLayoutBinding(VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_VERTEX_BIT, 0),
    };
    VkDescriptorSetLayoutCreateInfo descriptorLayout =
            vks::initializers::descriptorSetLayoutCreateInfo(setLayoutBindings);
    VK_CHECK_RESULT(vkCreateDescriptorSetLayout(device, &descriptorLayout, nullptr, &descriptorSetLayout));

    // Pipeline layout
    VkPipelineLayoutCreateInfo pPipelineLayoutCreateInfo = vks::initializers::pipelineLayoutCreateInfo(&descriptorSetLayout, 1);
    VK_CHECK_RESULT(vkCreatePipelineLayout(device, &pPipelineLayoutCreateInfo, nullptr, &pipelineLayout));

    // Descriptor set
    VkDescriptorSetAllocateInfo allocInfo =	vks::initializers::descriptorSetAllocateInfo(descriptorPool, &descriptorSetLayout, 1);
    VK_CHECK_RESULT(vkAllocateDescriptorSets(device, &allocInfo, &descriptorSet));
    std::vector<VkWriteDescriptorSet> writeDescriptorSets = {
        vks::initializers::writeDescriptorSet(descriptorSet, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 0, &uniformBufferVS.descriptor),
    };
    vkUpdateDescriptorSets(device, static_cast<uint32_t>(writeDescriptorSets.size()), writeDescriptorSets.data(), 0, nullptr);
}

void VulkanExample::preparePipelines()
{
    // Rendering
    VkPipelineInputAssemblyStateCreateInfo inputAssemblyState =
            vks::initializers::pipelineInputAssemblyStateCreateInfo(VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST, 0,	VK_FALSE);

    VkPipelineRasterizationStateCreateInfo rasterizationState =
            vks::initializers::pipelineRasterizationStateCreateInfo(VK_POLYGON_MODE_FILL, VK_CULL_MODE_FRONT_BIT, VK_FRONT_FACE_COUNTER_CLOCKWISE);

    VkPipelineColorBlendAttachmentState blendAttachmentState =
            vks::initializers::pipelineColorBlendAttachmentState(0xf, VK_FALSE);

    VkPipelineColorBlendStateCreateInfo colorBlendState =
            vks::initializers::pipelineColorBlendStateCreateInfo(1, &blendAttachmentState);

    VkPipelineDepthStencilStateCreateInfo depthStencilState =
            vks::initializers::pipelineDepthStencilStateCreateInfo(VK_TRUE, VK_TRUE, VK_COMPARE_OP_LESS_OR_EQUAL);

    VkPipelineViewportStateCreateInfo viewportState =
            vks::initializers::pipelineViewportStateCreateInfo(1, 1, 0);

    VkPipelineMultisampleStateCreateInfo multisampleState =
            vks::initializers::pipelineMultisampleStateCreateInfo(VK_SAMPLE_COUNT_1_BIT);

    std::vector<VkDynamicState> dynamicStateEnables = {
        VK_DYNAMIC_STATE_VIEWPORT,
        VK_DYNAMIC_STATE_SCISSOR
    };
    VkPipelineDynamicStateCreateInfo dynamicState =
            vks::initializers::pipelineDynamicStateCreateInfo(dynamicStateEnables);

    // Load shaders
    std::array<VkPipelineShaderStageCreateInfo,2> shaderStages;

    VkGraphicsPipelineCreateInfo pipelineCreateInfo = vks::initializers::pipelineCreateInfo(pipelineLayout, renderPass);

    pipelineCreateInfo.pInputAssemblyState = &inputAssemblyState;
    pipelineCreateInfo.pRasterizationState = &rasterizationState;
    pipelineCreateInfo.pColorBlendState = &colorBlendState;
    pipelineCreateInfo.pMultisampleState = &multisampleState;
    pipelineCreateInfo.pViewportState = &viewportState;
    pipelineCreateInfo.pDepthStencilState = &depthStencilState;
    pipelineCreateInfo.pDynamicState = &dynamicState;
    pipelineCreateInfo.stageCount = static_cast<uint32_t>(shaderStages.size());
    pipelineCreateInfo.pStages = shaderStages.data();

    std::vector<VkVertexInputBindingDescription> vertexInputBindings = {
        vks::initializers::vertexInputBindingDescription(0, vertexLayout.stride(), VK_VERTEX_INPUT_RATE_VERTEX),
    };
    std::vector<VkVertexInputAttributeDescription> vertexInputAttributes = {
        vks::initializers::vertexInputAttributeDescription(0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0),					// Location 0: Position
        vks::initializers::vertexInputAttributeDescription(0, 1, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 3),	// Location 1: Normal
        vks::initializers::vertexInputAttributeDescription(0, 2, VK_FORMAT_R32G32B32_SFLOAT, sizeof(float) * 6),	// Location 2: Color
    };
    VkPipelineVertexInputStateCreateInfo vertexInputState = vks::initializers::pipelineVertexInputStateCreateInfo();
    vertexInputState.vertexBindingDescriptionCount = static_cast<uint32_t>(vertexInputBindings.size());
    vertexInputState.pVertexBindingDescriptions = vertexInputBindings.data();
    vertexInputState.vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexInputAttributes.size());
    vertexInputState.pVertexAttributeDescriptions = vertexInputAttributes.data();

    pipelineCreateInfo.pVertexInputState = &vertexInputState;

    shaderStages[0] = Saiga::Vulkan::shaderLoader.loadShader(device, ASSET_PATH "shaders/imgui/scene.vert.spv", VK_SHADER_STAGE_VERTEX_BIT);
    shaderStages[1] = Saiga::Vulkan::shaderLoader.loadShader(device,ASSET_PATH "shaders/imgui/scene.frag.spv", VK_SHADER_STAGE_FRAGMENT_BIT);
    VK_CHECK_RESULT(vkCreateGraphicsPipelines(device, pipelineCache, 1, &pipelineCreateInfo, nullptr, &pipeline));
}

void VulkanExample::prepareUniformBuffers()
{
    // Vertex shader uniform buffer block
    VK_CHECK_RESULT(vulkanDevice->createBuffer(
                        VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                        &uniformBufferVS,
                        sizeof(uboVS),
                        &uboVS));

    updateUniformBuffers();
}

void VulkanExample::updateUniformBuffers()
{
    // Vertex shader
    uboVS.projection = camera.matrices.perspective;
    uboVS.modelview = camera.matrices.view * glm::mat4(1.0f);

    // Light source
    if (uiSettings.animateLight) {
        uiSettings.lightTimer += frameTimer * uiSettings.lightSpeed;
        uboVS.lightPos.x = sin(glm::radians(uiSettings.lightTimer * 360.0f)) * 15.0f;
        uboVS.lightPos.z = cos(glm::radians(uiSettings.lightTimer * 360.0f)) * 15.0f;
    };

    VK_CHECK_RESULT(uniformBufferVS.map());
    memcpy(uniformBufferVS.mapped, &uboVS, sizeof(uboVS));
    uniformBufferVS.unmap();
}

void VulkanExample::draw()
{
    VulkanExampleBase::prepareFrame();
    buildCommandBuffers();
    submitInfo.commandBufferCount = 1;
    submitInfo.pCommandBuffers = &drawCmdBuffers[currentBuffer];
    VK_CHECK_RESULT(vkQueueSubmit(queue, 1, &submitInfo, VK_NULL_HANDLE));
    VulkanExampleBase::submitFrame();
}

void VulkanExample::loadAssets()
{
    models.models.loadFromFile(ASSET_PATH "models/vulkanscenemodels.dae", vertexLayout, 1.0f, vulkanDevice, queue);
    models.background.loadFromFile(ASSET_PATH "models/vulkanscenebackground.dae", vertexLayout, 1.0f, vulkanDevice, queue);
    models.logos.loadFromFile(ASSET_PATH "models/vulkanscenelogos.dae", vertexLayout, 1.0f, vulkanDevice, queue);
}

void VulkanExample::prepareImGui()
{
    imGui = std::make_shared<ImGUI>(this);
    imGui->init((float)width, (float)height);
    imGui->initResources(renderPass, queue);
}

void VulkanExample::prepare()
{
    VulkanExampleBase::prepare();
    loadAssets();
    prepareUniformBuffers();
    setupLayoutsAndDescriptors();
    preparePipelines();
    prepareImGui();
    buildCommandBuffers();
    prepared = true;
}

void VulkanExample::render()
{
    if (!prepared)
        return;

    // Update imGui
    ImGuiIO& io = ImGui::GetIO();

    io.DisplaySize = ImVec2((float)width, (float)height);
    io.DeltaTime = frameTimer;

    io.MousePos = ImVec2(mousePos.x, mousePos.y);
    io.MouseDown[0] = mouseButtons.left;
    io.MouseDown[1] = mouseButtons.right;

    draw();

    if (uiSettings.animateLight)
        updateUniformBuffers();
}
