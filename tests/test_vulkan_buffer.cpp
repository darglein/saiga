/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/buffer/Buffer.h"

#include <numeric>

#include "vulkan_test_context.h"

using namespace Saiga;

class VulkanBuffer : public VulkanTest
{
   protected:
    static std::vector<int> Numbers(int n)
    {
        std::vector<int> data(n);
        std::iota(data.begin(), data.end(), 1);
        return data;
    }
};

TEST_F(VulkanBuffer, HostVisibleRoundTrip)
{
    auto data   = Numbers(1000);
    size_t size = data.size() * sizeof(int);

    Vulkan::Buffer buffer;
    buffer.createBuffer(base(), size, vk::BufferUsageFlagBits::eStorageBuffer,
                        vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    ASSERT_TRUE(bool(buffer));
    // The allocation can be larger than requested (alignment).
    EXPECT_GE(buffer.size(), size);

    buffer.upload(data.data(), size);

    std::vector<int> result(data.size(), 0);
    buffer.download(result.data(), size);
    EXPECT_EQ(result, data);
}

TEST_F(VulkanBuffer, StagedRoundTrip)
{
    auto data   = Numbers(4096);
    size_t size = data.size() * sizeof(int);

    Vulkan::Buffer buffer;
    buffer.createBuffer(base(), size,
                        vk::BufferUsageFlagBits::eStorageBuffer | vk::BufferUsageFlagBits::eTransferSrc |
                            vk::BufferUsageFlagBits::eTransferDst,
                        vk::MemoryPropertyFlagBits::eDeviceLocal);
    ASSERT_TRUE(bool(buffer));

    buffer.stagedUpload(base(), size, data.data());

    // stagedDownload copies the whole allocation.
    std::vector<char> result(buffer.size());
    buffer.stagedDownload(result.data());
    EXPECT_EQ(std::memcmp(result.data(), data.data(), size), 0);
}

TEST_F(VulkanBuffer, CopyBetweenBuffers)
{
    auto data   = Numbers(256);
    size_t size = data.size() * sizeof(int);

    auto usage  = vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst;
    auto memory = vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent;

    Vulkan::Buffer src, dst;
    src.createBuffer(base(), size, usage, memory);
    dst.createBuffer(base(), size, usage, memory);
    src.upload(data.data(), size);

    Submit(
        [&](vk::CommandBuffer cmd)
        {
            src.copyTo(cmd, dst, 0, 0, size);
            dst.barrier(cmd, vk::PipelineStageFlagBits2::eTransfer, vk::AccessFlagBits2::eTransferWrite,
                        vk::PipelineStageFlagBits2::eHost, vk::AccessFlagBits2::eHostRead);
        });

    std::vector<int> result(data.size(), 0);
    dst.download(result.data(), size);
    EXPECT_EQ(result, data);
}

// A moved-from buffer must not release the allocation of the new owner.
TEST_F(VulkanBuffer, Move)
{
    auto data   = Numbers(16);
    size_t size = data.size() * sizeof(int);

    Vulkan::Buffer a;
    a.createBuffer(base(), size, vk::BufferUsageFlagBits::eStorageBuffer,
                   vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    a.upload(data.data(), size);

    Vulkan::Buffer b = std::move(a);
    EXPECT_FALSE(bool(a));
    ASSERT_TRUE(bool(b));
    a.destroy();

    std::vector<int> result(data.size(), 0);
    b.download(result.data(), size);
    EXPECT_EQ(result, data);
}
