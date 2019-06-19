/*
 * Vulkan Example - imGui (https://github.com/ocornut/imgui)
 *
 * Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */

#include "compute.h"

#include "saiga/core/util/color.h"

#include <saiga/core/imgui/imgui.h>

#if defined(SAIGA_OPENGL_INCLUDED)
#    error OpenGL was included somewhere.
#endif

Compute::Compute(Saiga::Vulkan::VulkanWindow& window, Saiga::Vulkan::VulkanForwardRenderer& renderer)
    : Updating(window), Saiga::Vulkan::VulkanForwardRenderingInterface(renderer), renderer(renderer)
{
    float aspect = window.getAspectRatio();
    camera.setProj(60.0f, aspect, 0.1f, 50.0f, true);
    camera.setView(vec3(0, 5, 10), vec3(0, 0, 0), vec3(0, 1, 0));
    camera.rotationPoint = make_vec3(0);

    window.setCamera(&camera);

    init(renderer.base());
}

Compute::~Compute()
{
    assetRenderer.destroy();
    compute.storageBuffer.destroy();
    computePipeline.destroy();
    // compute.queue.destroy();
    compute.storageTexture.destroy();
}

void Compute::init(Saiga::Vulkan::VulkanBase& base)
{
    using namespace Saiga;

    {
        Saiga::TemplatedImage<float> od(100, 100);
        Saiga::Vulkan::Texture2D texture;
        texture.fromImage(base, od, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage);
    }

    {
        Saiga::Vulkan::Texture2D outTexture;
        TemplatedImage<ucvec4> img(100, 100);
        img.getImageView().set(ucvec4(0, 0, 255, 255));
        outTexture.fromImage(base, img, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage);

        vk::CommandBuffer cmd = base.mainQueue.commandPool.createAndBeginOneTimeBuffer();
        outTexture.transitionImageLayout(cmd, vk::ImageLayout::eGeneral);
        cmd.end();
        base.mainQueue.submitAndWait(cmd);
    }


    vulkanDevice = &renderer.base();
    device       = vulkanDevice->device;

    // create storage buffer
    compute.data.resize(10, 1);
    //    compute.storageBuffer.
    compute.storageBuffer.createBuffer(
        renderer.base(), sizeof(int) * compute.data.size(), vk::BufferUsageFlagBits::eStorageBuffer,
        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    //    compute.storageBuffer.allocateMemoryBuffer(renderer.base(),vk::MemoryPropertyFlagBits::eHostVisible|vk::MemoryPropertyFlagBits::eHostCoherent);
    compute.storageBuffer.upload(compute.data.data(), compute.data.size());


    {
        Saiga::TemplatedImage<ucvec4> img(100, 100);
        img.getImageView().set(ucvec4(0, 0, 255, 255));
        compute.storageTexture.fromImage(base, img,
                                         vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage);
        // compute.storageTexture.transitionImageLayout();

        vk::CommandBuffer cmd = base.mainQueue.commandPool.createAndBeginOneTimeBuffer();
        compute.storageTexture.transitionImageLayout(cmd, vk::ImageLayout::eGeneral);
        cmd.end();
        base.mainQueue.submitAndWait(cmd);
    }


    computePipeline.init(base, 1);
    computePipeline.addDescriptorSetLayout(
        {{0, {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}},
         {1, {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute}}});
    computePipeline.shaderPipeline.loadCompute(device, "vulkan/test.comp");
    computePipeline.create();



    descriptorSet = computePipeline.createDescriptorSet();

    vk::DescriptorBufferInfo descriptorInfo = compute.storageBuffer.createInfo();
    auto iinfo                              = compute.storageTexture.getDescriptorInfo();
    device.updateDescriptorSets(
        {
            vk::WriteDescriptorSet(descriptorSet, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &descriptorInfo,
                                   nullptr),
            vk::WriteDescriptorSet(descriptorSet, 1, 0, 1, vk::DescriptorType::eStorageImage, &iinfo, nullptr, nullptr),
        },
        nullptr);


    // compute.queue.create(device, vulkanDevice->queueFamilyIndices.compute);
    compute.commandBuffer = base.computeQueue->commandPool.allocateCommandBuffer();

    {
        // Build the command buffer
        compute.commandBuffer.begin(vk::CommandBufferBeginInfo(vk::CommandBufferUsageFlagBits::eOneTimeSubmit));
        if (computePipeline.bind(compute.commandBuffer))
        {
            computePipeline.bindDescriptorSet(compute.commandBuffer, descriptorSet);
            // Dispatch 1 block
            compute.commandBuffer.dispatch(1, 1, 1);
            compute.commandBuffer.end();
        }
    }


    base.computeQueue->submitAndWait(compute.commandBuffer);
    compute.storageBuffer.download(compute.data.data());

    for (int i : compute.data) std::cout << i << std::endl;
}



void Compute::update(float dt)
{
    camera.update(dt);
    camera.interpolate(dt, 0);
}


void Compute::render(vk::CommandBuffer cmd) {}

void Compute::renderGUI() {}

void Compute::keyPressed(SDL_Keysym key)
{
    switch (key.scancode)
    {
        case SDL_SCANCODE_ESCAPE:
            parentWindow.close();
            break;
        default:
            break;
    }
}

void Compute::keyReleased(SDL_Keysym key) {}
