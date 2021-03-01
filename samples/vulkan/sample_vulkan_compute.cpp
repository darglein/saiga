/*
 * Vulkan Example - imGui (https://github.com/ocornut/imgui)
 *
 * Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */



#include "saiga/core/sdl/sdl_camera.h"
#include "saiga/core/sdl/sdl_eventhandler.h"
#include "saiga/core/util/color.h"
#include "saiga/core/window/Interfaces.h"
#include "saiga/vulkan/CommandPool.h"
#include "saiga/vulkan/VulkanForwardRenderer.h"
#include "saiga/vulkan/buffer/Buffer.h"
#include "saiga/vulkan/pipeline/ComputePipeline.h"
#include "saiga/vulkan/pipeline/DescriptorSet.h"
#include "saiga/vulkan/renderModules/AssetRenderer.h"
#include "saiga/vulkan/texture/Texture.h"
#include "saiga/vulkan/window/SDLSample.h"

#include <saiga/core/imgui/imgui.h>
using namespace Saiga;

class Compute : public VulkanSDLExampleBase
{
   public:
    Compute();
    ~Compute();

    void init(Saiga::Vulkan::VulkanBase& base);

    void update(float dt) override;

    void render(vk::CommandBuffer cmd) override;



   private:
    Saiga::SDLCamera<Saiga::PerspectiveCamera> camera;

    Saiga::Object3D teapotTrans;
    Saiga::Vulkan::VulkanVertexColoredAsset teapot, plane;
    Saiga::Vulkan::AssetRenderer assetRenderer;


    Saiga::Vulkan::VulkanBase* vulkanDevice;
    vk::Device device;

    struct
    {  // Uniform buffer object containing particle system parameters
        std::vector<int> data;
        Saiga::Vulkan::Buffer storageBuffer;
        Saiga::Vulkan::Texture2D storageTexture;
        vk::CommandBuffer commandBuffer;  // Command buffer storing the dispatch commands and barriers
    } compute;
    Saiga::Vulkan::ComputePipeline computePipeline;
    Saiga::Vulkan::StaticDescriptorSet descriptorSet;
};


Compute::Compute()
{
    init(renderer->base());
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


    vulkanDevice = &renderer->base();
    device       = vulkanDevice->device;

    // create storage buffer
    compute.data.resize(10, 1);
    //    compute.storageBuffer.
    compute.storageBuffer.createBuffer(
        renderer->base(), sizeof(int) * compute.data.size(), vk::BufferUsageFlagBits::eStorageBuffer,
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
#undef main
int main(const int argc, const char* argv[])
{
    initSaigaSample();
    Compute example;
    example.run();
    return 0;
}
