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
