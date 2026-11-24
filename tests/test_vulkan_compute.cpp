/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/buffer/Buffer.h"
#include "saiga/vulkan/pipeline/ComputePipeline.h"
#include "saiga/vulkan/texture/Texture.h"

#include "vulkan_test_context.h"

using namespace Saiga;

class VulkanCompute : public VulkanTest
{
};

// vulkan/test.comp, for the invocations i < 10: numbers[i] = i and numbers[0] += 10 (atomic). It also writes white to
// the pixel (0,0) of the storage image.
TEST_F(VulkanCompute, StorageBufferAndImage)
{
    std::vector<uint32_t> data(10, 1);
    size_t size = data.size() * sizeof(uint32_t);

    Vulkan::Buffer storageBuffer;
    storageBuffer.createBuffer(base(), size, vk::BufferUsageFlagBits::eStorageBuffer,
                               vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    storageBuffer.upload(data.data(), size);

    Vulkan::Texture2D texture;
    {
        TemplatedImage<ucvec4> img(4, 4);
        img.getImageView().set(ucvec4(0, 0, 255, 255));
        texture.fromImage(base(), img, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage, false);
    }

    Vulkan::ComputePipeline pipeline;
    pipeline.init(base(), 1);
    pipeline.addDescriptorSetLayout(
        {{0, {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}},
         {1, {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute}}});
    ASSERT_TRUE(pipeline.shaderPipeline.loadCompute(base().device, "vulkan/test.comp"));
    pipeline.create();

    auto set = pipeline.createDescriptorSet();

    Vulkan::StagingBuffer pixels;
    pixels.init(base(), 4 * 4 * 4);

    bool bound = false;
    Submit(
        [&](vk::CommandBuffer cmd)
        {
            texture.transitionImageLayout(cmd, vk::ImageLayout::eGeneral);

            auto bufferInfo       = storageBuffer.createInfo();
            auto imageInfo        = texture.getDescriptorInfo();
            imageInfo.imageLayout = vk::ImageLayout::eGeneral;
            base().device.updateDescriptorSets(
                {
                    vk::WriteDescriptorSet(set, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &bufferInfo,
                                           nullptr),
                    vk::WriteDescriptorSet(set, 1, 0, 1, vk::DescriptorType::eStorageImage, &imageInfo, nullptr,
                                           nullptr),
                },
                nullptr);

            bound = pipeline.bind(cmd);
            if (bound)
            {
                pipeline.bindDescriptorSet(cmd, set);
                cmd.dispatch(1, 1, 1);
            }

            // Read the image back (the transition waits for the shader writes).
            texture.transitionImageLayout(cmd, vk::ImageLayout::eTransferSrcOptimal);
            vk::BufferImageCopy region = pixels.getBufferImageCopy(0);
            region.imageSubresource    = vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1);
            region.imageExtent         = vk::Extent3D(4, 4, 1);
            cmd.copyImageToBuffer(texture.memoryLocation->data.image, vk::ImageLayout::eTransferSrcOptimal,
                                  pixels.createInfo().buffer, region);

            // Make the shader and transfer writes visible to the host reads below.
            vk::MemoryBarrier2 toHost(
                vk::PipelineStageFlagBits2::eComputeShader | vk::PipelineStageFlagBits2::eAllTransfer,
                vk::AccessFlagBits2::eShaderStorageWrite | vk::AccessFlagBits2::eTransferWrite,
                vk::PipelineStageFlagBits2::eHost, vk::AccessFlagBits2::eHostRead);
            cmd.pipelineBarrier2(vk::DependencyInfo({}, toHost, nullptr, nullptr));
        });
    ASSERT_TRUE(bound);

    storageBuffer.download(data.data(), size);
    for (uint32_t i = 1; i < data.size(); ++i) EXPECT_EQ(data[i], i);
    // numbers[0] = 0 of invocation 0 is not ordered with the atomic adds of the other invocations, but its own add
    // comes after it.
    EXPECT_GE(data[0], 10u);
    EXPECT_LE(data[0], 100u);
    EXPECT_EQ(data[0] % 10, 0u);

    TemplatedImage<ucvec4> img(4, 4);
    pixels.download(img.data(), 4 * 4 * 4);
    pixels.destroyImmediately();
    EXPECT_EQ(img(0, 0), ucvec4(255, 255, 255, 255));
    EXPECT_EQ(img(1, 1), ucvec4(0, 0, 255, 255));
}

// A pipeline from the pipeline pool (VulkanBase::reusePipelines) creates no shader module: its key needs only the
// SPIR-V hashes (test speed, S36 review round 2: vkCreateShaderModule validates the SPIR-V with the validation layer,
// ~0.4 ms per lighting shader, which every new scene of a test paid for its pooled pipelines). A pipeline that is built
// creates its module, and a pooled pipeline still dispatches
TEST_F(VulkanCompute, PooledPipelineCreatesNoShaderModule)
{
    if (!base().reusePipelines) GTEST_SKIP() << "pipeline reuse is off (SAIGA_TEST_NEW_DEVICES=1)";
    auto make = [&](Vulkan::ComputePipeline& p)
    {
        p.init(base(), 1);
        p.addDescriptorSetLayout({{0, {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}},
                                  {1, {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute}}});
        ASSERT_TRUE(p.shaderPipeline.loadCompute(base().device, "vulkan/test.comp"));
        // Loaded and valid, the module not created yet
        EXPECT_TRUE(p.shaderPipeline.valid());
        ASSERT_NE(p.shaderPipeline.moduleAt(0), nullptr);
        EXPECT_FALSE(p.shaderPipeline.moduleAt(0)->module);
        EXPECT_GT(p.shaderPipeline.moduleAt(0)->spirvSize, 0u);
        p.create();
    };
    {
        Vulkan::ComputePipeline first;
        make(first);
        // (built here or taken from the pool of an earlier test: destroyed, it is in the pool for the next one)
        first.destroy();
        // (the pipeline returns to the pool when the device is done with it)
        base().flushDeferredDeletes();
    }
    const auto before = base().pipelineReuseStats();
    Vulkan::ComputePipeline second;
    make(second);
    const auto after = base().pipelineReuseStats();
    EXPECT_EQ(after.hits, before.hits + 1);
    EXPECT_EQ(after.misses, before.misses);
    EXPECT_FALSE(second.shaderPipeline.moduleAt(0)->module);
    // A pooled pipeline binds and dispatches (validation clean)
    Vulkan::Buffer storage;
    storage.createBuffer(base(), 64, vk::BufferUsageFlagBits::eStorageBuffer,
                         vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    Vulkan::Texture2D texture;
    TemplatedImage<ucvec4> img(4, 4);
    img.getImageView().set(ucvec4(0, 0, 255, 255));
    texture.fromImage(base(), img, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage, false);
    auto set   = second.createDescriptorSet();
    bool bound = false;
    Submit(
        [&](vk::CommandBuffer cmd)
        {
            texture.transitionImageLayout(cmd, vk::ImageLayout::eGeneral);
            auto bufferInfo       = storage.createInfo();
            auto imageInfo        = texture.getDescriptorInfo();
            imageInfo.imageLayout = vk::ImageLayout::eGeneral;
            base().device.updateDescriptorSets(
                {vk::WriteDescriptorSet(set, 0, 0, 1, vk::DescriptorType::eStorageBuffer, nullptr, &bufferInfo),
                 vk::WriteDescriptorSet(set, 1, 0, 1, vk::DescriptorType::eStorageImage, &imageInfo, nullptr)},
                nullptr);
            bound = second.bind(cmd);
            if (bound)
            {
                second.bindDescriptorSet(cmd, set);
                cmd.dispatch(1, 1, 1);
            }
        });
    EXPECT_TRUE(bound);
    // A pipeline built from the shader (a reload never uses the pool) creates the module
    second.reload();
    EXPECT_TRUE(second.shaderPipeline.moduleAt(0)->module);
}
