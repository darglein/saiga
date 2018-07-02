/*
* Vulkan Example - imGui (https://github.com/ocornut/imgui)
*
* Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
*
* This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
*/

#include "sample.h"

UISettings uiSettings;

VulkanExample::VulkanExample()
    : VulkanExampleBase(ENABLE_VALIDATION)
{
    title = "Vulkan Example - ImGui";
    camera.type = Camera::CameraType::lookat;
    camera.setPosition(glm::vec3(0.0f, 1.4f, -4.8f));
    camera.setRotation(glm::vec3(4.5f, -380.0f, 0.0f));
    camera.setPerspective(45.0f, (float)width / (float)height, 0.1f, 256.0f);
}

VulkanExample::~VulkanExample()
{
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

    ImGui::NewFrame();


    ImGui::SetNextWindowSize(ImVec2(200, 200), ImGuiSetCond_FirstUseEver);
    ImGui::Begin("Example settings");
    ImGui::Checkbox("Render models", &uiSettings.displayModels);
    ImGui::Checkbox("Display logos", &uiSettings.displayLogos);
    ImGui::Checkbox("Display background", &uiSettings.displayBackground);
    ImGui::Checkbox("Animate light", &uiSettings.animateLight);
    ImGui::SliderFloat("Light speed", &uiSettings.lightSpeed, 0.1f, 1.0f);
    ImGui::End();

    ImGui::SetNextWindowPos(ImVec2(650, 20), ImGuiSetCond_FirstUseEver);
    ImGui::ShowTestWindow();

    // Render to generate draw buffers
    ImGui::Render();

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
        assetRenderer.bind(drawCmdBuffers[i]);
        //        vkCmdBindDescriptorSets(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipelineLayout, 0, 1, &descriptorSet, 0, nullptr);
        //        vkCmdBindPipeline(drawCmdBuffers[i], VK_PIPELINE_BIND_POINT_GRAPHICS, pipeline);

        VkDeviceSize offsets[1] = { 0 };

#if 0
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
#endif

        vkCmdBindVertexBuffers(drawCmdBuffers[i], 0, 1, &teapot.vertices.buffer, offsets);
        vkCmdBindIndexBuffer(drawCmdBuffers[i], teapot.indices.buffer, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(drawCmdBuffers[i], teapot.indexCount, 1, 0, 0, 0);

        // Render imGui
        imGui->drawFrame(drawCmdBuffers[i]);

        vkCmdEndRenderPass(drawCmdBuffers[i]);

        VK_CHECK_RESULT(vkEndCommandBuffer(drawCmdBuffers[i]));
    }
}

void VulkanExample::updateUniformBuffers()
{
    // Vertex shader
    //    uboVS.projection = camera.matrices.perspective;
    //    uboVS.modelview = camera.matrices.view * glm::mat4(1.0f);
    assetRenderer.updateUniformBuffers(camera.matrices.view,camera.matrices.perspective);
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
//    models.models.loadFromFile(ASSET_PATH "models/vulkanscenemodels.dae", Saiga::Vulkan::AssetRenderer::vertexLayout, 1.0f, vulkanDevice, queue);
//    models.background.loadFromFile(ASSET_PATH "models/vulkanscenebackground.dae", Saiga::Vulkan::AssetRenderer::vertexLayout, 1.0f, vulkanDevice, queue);
//    models.logos.loadFromFile(ASSET_PATH "models/vulkanscenelogos.dae", Saiga::Vulkan::AssetRenderer::vertexLayout, 1.0f, vulkanDevice, queue);


    teapot.load("objs/teapot.obj", vulkanDevice, queue);


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
    assetRenderer.prepareUniformBuffers(vulkanDevice);
    updateUniformBuffers();
    assetRenderer.setupLayoutsAndDescriptors(device);
    assetRenderer.preparePipelines(device,pipelineCache,renderPass);
    prepareImGui();
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
