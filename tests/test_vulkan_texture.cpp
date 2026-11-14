/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/buffer/StagingBuffer.h"
#include "saiga/vulkan/texture/Texture.h"

#include "vulkan_test_context.h"

using namespace Saiga;

class VulkanTexture : public VulkanTest
{
   protected:
    static TemplatedImage<ucvec4> Pattern(int w, int h)
    {
        TemplatedImage<ucvec4> img(h, w);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) img(y, x) = ucvec4(x * 7, y * 13, x + y, 255 - x);
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

    static void ExpectEqual(const TemplatedImage<ucvec4>& a, const TemplatedImage<ucvec4>& b)
    {
        ASSERT_EQ(a.w, b.w);
        ASSERT_EQ(a.h, b.h);
        for (int y = 0; y < a.h; ++y)
            for (int x = 0; x < a.w; ++x) ASSERT_EQ(a(y, x), b(y, x)) << "at " << x << "," << y;
    }
};

TEST_F(VulkanTexture, UploadDownload)
{
    auto img = Pattern(20, 12);

    Vulkan::Texture2D texture;
    texture.fromImage(base(), img, vk::ImageUsageFlagBits::eSampled, false);
    EXPECT_EQ(texture.width, 20u);
    EXPECT_EQ(texture.height, 12u);
    EXPECT_EQ(texture.format, vk::Format::eR8G8B8A8Unorm);

    ExpectEqual(Download(texture), img);
}

TEST_F(VulkanTexture, FlipY)
{
    auto img = Pattern(8, 6);

    Vulkan::Texture2D texture;
    texture.fromImage(base(), img, vk::ImageUsageFlagBits::eSampled, true);

    auto result = Download(texture);
    for (int y = 0; y < img.h; ++y)
        for (int x = 0; x < img.w; ++x) ASSERT_EQ(result(y, x), img(img.h - 1 - y, x)) << "at " << x << "," << y;
}

// 3-channel images are expanded to 4 channels, because the 3-channel formats are almost never supported.
TEST_F(VulkanTexture, RGBIsExpandedToRGBA)
{
    TemplatedImage<ucvec3> img(5, 9);
    for (int y = 0; y < img.h; ++y)
        for (int x = 0; x < img.w; ++x) img(y, x) = ucvec3(x * 20, y * 40, 3);

    Vulkan::Texture2D texture;
    texture.fromImage(base(), img, vk::ImageUsageFlagBits::eSampled, false);
    EXPECT_EQ(texture.format, vk::Format::eR8G8B8A8Unorm);

    auto result = Download(texture);
    for (int y = 0; y < img.h; ++y)
        for (int x = 0; x < img.w; ++x)
            ASSERT_EQ(result(y, x), ucvec4(x * 20, y * 40, 3, 255)) << "at " << x << "," << y;
}

TEST_F(VulkanTexture, UploadImageReplacesContent)
{
    Vulkan::Texture2D texture;
    TemplatedImage<ucvec4> black(6, 6);
    black.getImageView().set(ucvec4(0, 0, 0, 255));
    texture.fromImage(base(), black, vk::ImageUsageFlagBits::eSampled, false);

    auto img = Pattern(6, 6);
    texture.uploadImage(img, false);
    ExpectEqual(Download(texture), img);
}
