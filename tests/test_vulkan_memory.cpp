/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/buffer/Buffer.h"
#include "saiga/vulkan/buffer/StagingBuffer.h"
#include "saiga/vulkan/pipeline/ComputePipeline.h"
#include "saiga/vulkan/renderModules/TextureDisplay.h"
#include "saiga/vulkan/texture/Texture.h"

#include <memory>
#include <numeric>

#include "vulkan_test_context.h"

using namespace Saiga;

// The memory allocator (VMA) and its defragmentation, see Vulkan::Memory::VulkanMemory.
class VulkanMemory : public VulkanTest
{
   protected:
    struct TestBuffer
    {
        std::unique_ptr<Vulkan::Buffer> buffer;
        uint32_t first = 0;
    };

    static Vulkan::Memory::VulkanMemory& Memory() { return base().memory; }

    // A device local buffer with the content first, first + 1, ...
    static TestBuffer CreateBuffer(size_t size, uint32_t first, bool dynamic)
    {
        TestBuffer result;
        result.first  = first;
        result.buffer = std::make_unique<Vulkan::Buffer>();
        result.buffer->createBuffer(base(), size, vk::BufferUsageFlagBits::eStorageBuffer,
                                    vk::MemoryPropertyFlagBits::eDeviceLocal);
        std::vector<uint32_t> data(size / sizeof(uint32_t));
        std::iota(data.begin(), data.end(), first);
        result.buffer->stagedUpload(base(), size, data.data());
        if (dynamic)
        {
            EXPECT_TRUE(result.buffer->mark_dynamic());
        }
        return result;
    }

    static void ExpectContent(TestBuffer& b)
    {
        std::vector<uint32_t> data(b.buffer->size() / sizeof(uint32_t));
        b.buffer->stagedDownload(data.data());
        for (size_t i = 0; i < data.size(); ++i)
        {
            ASSERT_EQ(data[i], b.first + i) << "at " << i;
        }
    }

    // Many dynamic buffers, every second one is freed.
    static std::vector<TestBuffer> FragmentedBuffers(int count, size_t size)
    {
        std::vector<TestBuffer> buffers;
        for (int i = 0; i < count; ++i) buffers.push_back(CreateBuffer(size, i * 1000, true));
        std::vector<TestBuffer> kept;
        for (int i = 0; i < count; ++i)
        {
            if (i % 2 == 0) kept.push_back(std::move(buffers[i]));
        }
        buffers.clear();
        // Runs the deferred deallocations.
        base().flushDeferredDeletes();
        return kept;
    }

    static TemplatedImage<ucvec4> Pattern(int w, int h, int seed)
    {
        TemplatedImage<ucvec4> img(h, w);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) img(y, x) = ucvec4(x * 7 + seed, y * 13, x + y + seed, 255 - x);
        return img;
    }

    // Reads back an RGBA8 texture.
    static TemplatedImage<ucvec4> Download(Vulkan::Texture2D& texture)
    {
        size_t size = size_t(texture.width) * texture.height * 4;
        Vulkan::StagingBuffer staging;
        staging.init(base(), size);

        Submit(
            [&](vk::CommandBuffer cmd)
            {
                texture.transitionImageLayout(cmd, vk::ImageLayout::eTransferSrcOptimal);
                vk::BufferImageCopy region = staging.getBufferImageCopy(0);
                region.imageSubresource    = vk::ImageSubresourceLayers(vk::ImageAspectFlagBits::eColor, 0, 0, 1);
                region.imageExtent         = vk::Extent3D(texture.width, texture.height, 1);
                cmd.copyImageToBuffer(texture.memoryLocation->data.image, vk::ImageLayout::eTransferSrcOptimal,
                                      staging.createInfo().buffer, region);
                staging.barrier(cmd, vk::PipelineStageFlagBits2::eAllTransfer, vk::AccessFlagBits2::eTransferWrite,
                                vk::PipelineStageFlagBits2::eHost, vk::AccessFlagBits2::eHostRead);
                texture.transitionImageLayout(cmd, vk::ImageLayout::eShaderReadOnlyOptimal);
            });

        TemplatedImage<ucvec4> img(texture.height, texture.width);
        staging.download(img.data(), size);
        staging.destroyImmediately();
        return img;
    }
};

TEST_F(VulkanMemory, HostVisibleBufferIsMapped)
{
    Vulkan::Buffer buffer;
    buffer.createBuffer(base(), 100, vk::BufferUsageFlagBits::eStorageBuffer,
                        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    // Every buffer has its own allocation of the requested size.
    EXPECT_EQ(buffer.size(), 100u);
    EXPECT_TRUE(buffer.isMapped());

    std::vector<char> data(100), result(100);
    std::iota(data.begin(), data.end(), 0);
    buffer.upload(data.data(), data.size());
    buffer.flush(base());
    buffer.download(result.data());
    EXPECT_EQ(result, data);

    // Mapped buffers stay static: the mapped pointer would change with a move.
    EXPECT_FALSE(buffer.mark_dynamic());
}

TEST_F(VulkanMemory, MarkDynamicKeepsContent)
{
    auto b            = CreateBuffer(64 * 1024, 17, false);
    vk::Buffer before = b.buffer->createInfo().buffer;

    EXPECT_TRUE(b.buffer->mark_dynamic());
    // The buffer was copied into the defragmentable pool.
    EXPECT_NE(b.buffer->createInfo().buffer, before);
    EXPECT_GE(Memory().dynamicMemoryUsage().second, 64u * 1024);
    ExpectContent(b);
}

TEST_F(VulkanMemory, FullDefragBuffers)
{
    auto statsBefore = Memory().getDefragmentationStats();
    auto buffers     = FragmentedBuffers(64, 256 * 1024);

    std::vector<vk::Buffer> handles;
    for (auto& b : buffers) handles.push_back(b.buffer->createInfo().buffer);

    Memory().full_defrag();

    auto stats = Memory().getDefragmentationStats();
    EXPECT_GT(stats.allocationsMoved, statsBefore.allocationsMoved);

    int moved = 0;
    for (size_t i = 0; i < buffers.size(); ++i)
    {
        if (buffers[i].buffer->createInfo().buffer != handles[i]) moved++;
        ExpectContent(buffers[i]);
    }
    EXPECT_GT(moved, 0);

    // Nothing is left to move.
    statsBefore = Memory().getDefragmentationStats();
    Memory().full_defrag();
    EXPECT_EQ(Memory().getDefragmentationStats().allocationsMoved, statsBefore.allocationsMoved);
}

TEST_F(VulkanMemory, FullDefragTextures)
{
    auto statsBefore = Memory().getDefragmentationStats();

    std::vector<std::unique_ptr<Vulkan::Texture2D>> textures;
    for (int i = 0; i < 32; ++i)
    {
        auto texture = std::make_unique<Vulkan::Texture2D>();
        texture->fromImage(base(), Pattern(128, 64, i), vk::ImageUsageFlagBits::eSampled, false);
        ASSERT_TRUE(texture->mark_dynamic());
        textures.push_back(std::move(texture));
    }
    std::vector<std::unique_ptr<Vulkan::Texture2D>> kept;
    std::vector<int> seeds;
    for (int i = 0; i < 32; ++i)
    {
        if (i % 2 == 0)
        {
            kept.push_back(std::move(textures[i]));
            seeds.push_back(i);
        }
    }
    textures.clear();
    base().flushDeferredDeletes();

    Memory().full_defrag();
    EXPECT_GT(Memory().getDefragmentationStats().allocationsMoved, statsBefore.allocationsMoved);

    for (size_t i = 0; i < kept.size(); ++i)
    {
        // The move keeps the layout.
        EXPECT_EQ(kept[i]->memoryLocation->data.layout, vk::ImageLayout::eShaderReadOnlyOptimal);
        auto img      = Download(*kept[i]);
        auto expected = Pattern(128, 64, seeds[i]);
        for (int y = 0; y < img.h; ++y)
            for (int x = 0; x < img.w; ++x) ASSERT_EQ(img(y, x), expected(y, x)) << "texture " << i;
    }
}

// A DynamicDescriptorSet follows a moved texture (new image, view and sampler).
TEST_F(VulkanMemory, MovedTextureIsRendered)
{
    // The filler is freed, so that the texture behind it is moved.
    auto filler = std::make_unique<Vulkan::Texture2D>();
    filler->fromImage(base(), Pattern(256, 256, 0), vk::ImageUsageFlagBits::eSampled, false);
    ASSERT_TRUE(filler->mark_dynamic());

    Vulkan::Texture2D texture;
    {
        TemplatedImage<ucvec4> img(4, 4);
        img.getImageView().set(ucvec4(255, 0, 0, 255));
        texture.fromImage(base(), img);
    }
    ASSERT_TRUE(texture.mark_dynamic());

    Vulkan::TextureDisplay display;
    display.init(base(), renderer->renderingFormats());
    auto set = display.createDynamicDescriptorSet();
    set.assign(0, &texture);
    app.onRender = [&](vk::CommandBuffer cmd)
    {
        if (display.bind(cmd)) display.renderTexture(cmd, set, vec2(8, 4), vec2(16, 16));
    };

    ucvec4 red(255, 0, 0, 255);
    EXPECT_EQ(RenderFrame()(12, 16), red);

    filler.reset();
    renderer->waitIdle();
    base().flushDeferredDeletes();

    vk::Image before = texture.memoryLocation->data.image;
    Memory().full_defrag();
    ASSERT_NE(texture.memoryLocation->data.image, before) << "The texture was not moved";

    auto img = RenderFrame();
    EXPECT_EQ(img(4, 8), red);
    EXPECT_EQ(img(12, 16), red);
    EXPECT_EQ(img(19, 23), red);
    EXPECT_EQ(img(40, 50), ClearColor());
}

// A DynamicDescriptorSet follows a moved storage buffer (new vk::Buffer). vulkan/test.comp writes numbers[i] = i for
// i in [1, 10).
TEST_F(VulkanMemory, MovedBufferInDescriptorSet)
{
    // The filler is freed, so that the buffer behind it is moved.
    auto filler = CreateBuffer(1024 * 1024, 0, true);
    auto b      = CreateBuffer(10 * sizeof(uint32_t), 1000, true);

    Vulkan::Texture2D texture;
    {
        TemplatedImage<ucvec4> img(4, 4);
        img.getImageView().set(ucvec4(0, 0, 255, 255));
        texture.fromImage(base(), img, vk::ImageUsageFlagBits::eSampled | vk::ImageUsageFlagBits::eStorage, false);
    }
    Submit([&](vk::CommandBuffer cmd) { texture.transitionImageLayout(cmd, vk::ImageLayout::eGeneral); });

    Vulkan::ComputePipeline pipeline;
    pipeline.init(base(), 1);
    pipeline.addDescriptorSetLayout(
        {{0, {0, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute}},
         {1, {1, vk::DescriptorType::eStorageImage, 1, vk::ShaderStageFlagBits::eCompute}}});
    ASSERT_TRUE(pipeline.shaderPipeline.loadCompute(base().device, "vulkan/test.comp"));
    pipeline.create();

    auto set = pipeline.createDynamicDescriptorSet();
    set.assign(0, b.buffer.get());
    set.assign(1, &texture);

    filler.buffer.reset();
    base().flushDeferredDeletes();
    vk::Buffer before = b.buffer->createInfo().buffer;
    Memory().full_defrag();
    ASSERT_NE(b.buffer->createInfo().buffer, before) << "The buffer was not moved";

    bool bound = false;
    Submit(
        [&](vk::CommandBuffer cmd)
        {
            bound = pipeline.bind(cmd);
            if (bound)
            {
                // Writes the new buffer into a new set.
                pipeline.bindDescriptorSet(cmd, set);
                cmd.dispatch(1, 1, 1);
            }
        });
    ASSERT_TRUE(bound);

    std::vector<uint32_t> data(10);
    b.buffer->stagedDownload(data.data());
    for (uint32_t i = 1; i < data.size(); ++i) EXPECT_EQ(data[i], i);
}

// A transition with an own barrier is reported with setImageLayout(): the move copies the image from that layout and
// keeps it.
TEST_F(VulkanMemory, MovedTextureKeepsReportedLayout)
{
    auto filler = std::make_unique<Vulkan::Texture2D>();
    filler->fromImage(base(), Pattern(256, 256, 0), vk::ImageUsageFlagBits::eSampled, false);
    ASSERT_TRUE(filler->mark_dynamic());

    Vulkan::Texture2D texture;
    texture.fromImage(base(), Pattern(32, 16, 5), vk::ImageUsageFlagBits::eSampled, false);
    ASSERT_TRUE(texture.mark_dynamic());

    Submit(
        [&](vk::CommandBuffer cmd)
        {
            vk::ImageMemoryBarrier2 barrier(
                vk::PipelineStageFlagBits2::eAllCommands, {}, vk::PipelineStageFlagBits2::eAllCommands,
                vk::AccessFlagBits2::eMemoryRead, vk::ImageLayout::eShaderReadOnlyOptimal, vk::ImageLayout::eGeneral,
                VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, texture.memoryLocation->data.image,
                vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1));
            cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, barrier));
        });
    texture.setImageLayout(vk::ImageLayout::eGeneral);

    filler.reset();
    base().flushDeferredDeletes();
    vk::Image before = texture.memoryLocation->data.image;
    Memory().full_defrag();
    ASSERT_NE(texture.memoryLocation->data.image, before) << "The texture was not moved";

    EXPECT_EQ(texture.getImageLayout(), vk::ImageLayout::eGeneral);
    auto img      = Download(texture);
    auto expected = Pattern(32, 16, 5);
    for (int y = 0; y < img.h; ++y)
        for (int x = 0; x < img.w; ++x) ASSERT_EQ(img(y, x), expected(y, x));
}

// A move copies all mip levels and layers.
TEST_F(VulkanMemory, MovedTextureArrayWithMipmaps)
{
    auto filler = std::make_unique<Vulkan::Texture2D>();
    filler->fromImage(base(), Pattern(256, 256, 0), vk::ImageUsageFlagBits::eSampled, false);
    ASSERT_TRUE(filler->mark_dynamic());

    std::vector<TemplatedImage<ucvec4>> layers = {Pattern(32, 32, 1), Pattern(32, 32, 2), Pattern(32, 32, 3)};
    std::vector<const Image*> images;
    for (auto& l : layers) images.push_back(&l);

    Vulkan::Texture2DArray texture;
    texture.fromImages(base(), images, vk::ImageUsageFlagBits::eSampled, false, true);
    ASSERT_EQ(texture.mipLevels, 6u);
    ASSERT_TRUE(texture.mark_dynamic());

    std::vector<std::vector<uint8_t>> before;
    for (uint32_t layer = 0; layer < 3; ++layer)
        for (uint32_t mip = 0; mip < texture.mipLevels; ++mip) before.push_back(texture.download(mip, layer));

    filler.reset();
    base().flushDeferredDeletes();
    vk::Image old = texture.memoryLocation->data.image;
    Memory().full_defrag();
    ASSERT_NE(texture.memoryLocation->data.image, old) << "The texture was not moved";

    size_t i = 0;
    for (uint32_t layer = 0; layer < 3; ++layer)
        for (uint32_t mip = 0; mip < texture.mipLevels; ++mip, ++i)
            EXPECT_EQ(texture.download(mip, layer), before[i]) << "layer " << layer << " mip " << mip;
}

TEST_F(VulkanMemory, IncrementalDefrag)
{
    auto statsBefore = Memory().getDefragmentationStats();
    auto buffers     = FragmentedBuffers(64, 256 * 1024);

    Memory().enableDefragmentation = true;
    Memory().defragBytesPerFrame   = 1024 * 1024;
    // A pass per frame, each moves at most 1 MiB.
    for (int i = 0; i < 100; ++i) RenderFrame();
    Memory().enableDefragmentation = false;
    renderer->waitIdle();

    auto stats = Memory().getDefragmentationStats();
    EXPECT_GT(stats.allocationsMoved, statsBefore.allocationsMoved);
    for (auto& b : buffers) ExpectContent(b);
}

// Regression: a location that is part of an incremental pass in flight must not be freed before the pass ends (VMA
// frees it with the operation DESTROY then).
TEST_F(VulkanMemory, DeallocateDuringIncrementalPass)
{
    auto buffers   = FragmentedBuffers(64, 256 * 1024);
    auto usedStart = Memory().dynamicMemoryUsage().second;
    EXPECT_GT(usedStart, 0u);

    Memory().enableDefragmentation = true;
    Memory().defragBytesPerFrame   = 1024 * 1024;
    // The frame submits the first pass, which ends in the next frame. Free all buffers in between.
    RenderFrame();
    renderer->waitIdle();
    for (auto& b : buffers) b.buffer->destroyImmediately();
    buffers.clear();

    for (int i = 0; i < 5; ++i) RenderFrame();
    Memory().enableDefragmentation = false;
    renderer->waitIdle();
    base().flushDeferredDeletes();

    // Everything was freed (the other tests of this suite have freed their buffers as well).
    EXPECT_EQ(Memory().dynamicMemoryUsage().second, 0u);
}
