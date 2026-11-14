/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/texture/Texture.h"

#include <cstring>

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

    // Reads back one mip level of one layer of an RGBA8 texture.
    static TemplatedImage<ucvec4> Download(Vulkan::Texture& texture, uint32_t mipLevel = 0, uint32_t layer = 0)
    {
        auto data = texture.download(mipLevel, layer);
        TemplatedImage<ucvec4> img(texture.mipHeight(mipLevel), texture.mipWidth(mipLevel));
        EXPECT_EQ(data.size(), size_t(img.w) * img.h * 4);
        if (data.size() == size_t(img.w) * img.h * 4) std::memcpy(img.data(), data.data(), data.size());
        return img;
    }

    static TemplatedImage<ucvec4> Solid(int w, int h, ucvec4 color)
    {
        TemplatedImage<ucvec4> img(h, w);
        img.getImageView().set(color);
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

TEST_F(VulkanTexture, DownloadKeepsLayout)
{
    auto img = Pattern(4, 4);
    Vulkan::Texture2D texture;
    texture.fromImage(base(), img, vk::ImageUsageFlagBits::eSampled, false);
    ASSERT_EQ(texture.memoryLocation->data.layout, vk::ImageLayout::eShaderReadOnlyOptimal);
    ExpectEqual(Download(texture), img);
    EXPECT_EQ(texture.memoryLocation->data.layout, vk::ImageLayout::eShaderReadOnlyOptimal);
}

TEST_F(VulkanTexture, FullMipLevels)
{
    EXPECT_EQ(Vulkan::Texture::fullMipLevels(1, 1), 1u);
    EXPECT_EQ(Vulkan::Texture::fullMipLevels(8, 8), 4u);
    EXPECT_EQ(Vulkan::Texture::fullMipLevels(9, 3), 4u);
    EXPECT_EQ(Vulkan::Texture::fullMipLevels(4, 4, 32), 6u);
}

TEST_F(VulkanTexture, Mipmaps)
{
    // Left half black, right half white: the average is gray.
    TemplatedImage<ucvec4> img(8, 8);
    for (int y = 0; y < img.h; ++y)
        for (int x = 0; x < img.w; ++x) img(y, x) = x < 4 ? ucvec4(0, 0, 0, 255) : ucvec4(255, 255, 255, 255);

    Vulkan::Texture2D texture;
    texture.fromImage(base(), img, vk::ImageUsageFlagBits::eSampled, false, true);
    ASSERT_EQ(texture.mipLevels, 4u);
    // The defragmentation copies all mip levels.
    EXPECT_TRUE(texture.is_movable());

    ExpectEqual(Download(texture, 0), img);

    // 2x2: still a sharp edge
    auto level2 = Download(texture, 2);
    ASSERT_EQ(level2.w, 2);
    ASSERT_EQ(level2.h, 2);
    EXPECT_EQ(level2(0, 0), ucvec4(0, 0, 0, 255));
    EXPECT_EQ(level2(1, 1), ucvec4(255, 255, 255, 255));

    auto level3 = Download(texture, 3);
    ASSERT_EQ(level3.w, 1);
    ASSERT_EQ(level3.h, 1);
    EXPECT_NEAR(level3(0, 0)(0), 128, 2);
    EXPECT_NEAR(level3(0, 0)(1), 128, 2);
    EXPECT_EQ(level3(0, 0)(3), 255);
}

// uploadImage must regenerate the mip levels from the new content.
TEST_F(VulkanTexture, UploadRegeneratesMipmaps)
{
    Vulkan::Texture2D texture;
    texture.fromImage(base(), Solid(4, 4, ucvec4(255, 0, 0, 255)), vk::ImageUsageFlagBits::eSampled, false, true);
    ASSERT_EQ(texture.mipLevels, 3u);
    EXPECT_EQ(Download(texture, 2)(0, 0), ucvec4(255, 0, 0, 255));

    texture.uploadImage(Solid(4, 4, ucvec4(0, 0, 255, 255)), false);
    EXPECT_EQ(Download(texture, 1)(1, 1), ucvec4(0, 0, 255, 255));
    EXPECT_EQ(Download(texture, 2)(0, 0), ucvec4(0, 0, 255, 255));
}

TEST_F(VulkanTexture, Array)
{
    std::vector<TemplatedImage<ucvec4>> layers;
    layers.push_back(Pattern(6, 4));
    layers.push_back(Solid(6, 4, ucvec4(10, 20, 30, 40)));
    layers.push_back(Solid(6, 4, ucvec4(200, 100, 50, 255)));

    Vulkan::Texture2DArray texture;
    texture.fromImages(base(), {&layers[0], &layers[1], &layers[2]}, vk::ImageUsageFlagBits::eSampled, false);
    EXPECT_EQ(texture.layerCount, 3u);
    EXPECT_EQ(texture.width, 6u);
    EXPECT_EQ(texture.height, 4u);
    for (uint32_t i = 0; i < 3; ++i) ExpectEqual(Download(texture, 0, i), layers[i]);

    // Replacing one layer keeps the others.
    auto replacement = Solid(6, 4, ucvec4(1, 2, 3, 4));
    texture.uploadLayer(1, replacement, false);
    ExpectEqual(Download(texture, 0, 0), layers[0]);
    ExpectEqual(Download(texture, 0, 1), replacement);
    ExpectEqual(Download(texture, 0, 2), layers[2]);
}

TEST_F(VulkanTexture, Cube)
{
    std::vector<TemplatedImage<ucvec4>> faces;
    std::vector<const Image*> pointers;
    for (int i = 0; i < 6; ++i) faces.push_back(Solid(4, 4, ucvec4(i * 40, 255 - i * 40, i, 255)));
    for (auto& f : faces) pointers.push_back(&f);

    Vulkan::TextureCube texture;
    texture.fromImages(base(), pointers, vk::ImageUsageFlagBits::eSampled, false, true);
    EXPECT_EQ(texture.layerCount, 6u);
    EXPECT_EQ(texture.mipLevels, 3u);
    for (uint32_t i = 0; i < 6; ++i)
    {
        ExpectEqual(Download(texture, 0, i), faces[i]);
        // The mipmaps are generated per face.
        EXPECT_EQ(Download(texture, 2, i)(0, 0), faces[i](0, 0));
    }

    auto replacement = Solid(4, 4, ucvec4(9, 8, 7, 255));
    texture.uploadFace(3, replacement, false);
    ExpectEqual(Download(texture, 0, 3), replacement);
    ExpectEqual(Download(texture, 0, 2), faces[2]);
    EXPECT_EQ(Download(texture, 2, 3)(0, 0), ucvec4(9, 8, 7, 255));
}

TEST_F(VulkanTexture, Volume)
{
    const int w = 5, h = 4, d = 3;
    std::vector<uint8_t> data(w * h * d);
    for (size_t i = 0; i < data.size(); ++i) data[i] = uint8_t(i * 3);

    Vulkan::Texture3D texture;
    texture.fromData(base(), w, h, d, vk::Format::eR8Unorm, data.data());
    EXPECT_EQ(texture.depth, 3u);
    EXPECT_EQ(texture.download(), data);

    // Replace a 2x2x2 box at (1,1,1)
    std::vector<uint8_t> box(8, 255);
    texture.uploadRegion(ivec3(1, 1, 1), ivec3(2, 2, 2), box.data());

    auto expected = data;
    for (int z = 1; z < 3; ++z)
        for (int y = 1; y < 3; ++y)
            for (int x = 1; x < 3; ++x) expected[(z * h + y) * w + x] = 255;
    EXPECT_EQ(texture.download(), expected);
}

TEST_F(VulkanTexture, VolumeMipmaps)
{
    std::vector<uint8_t> data(8 * 8 * 8, 77);
    Vulkan::Texture3D texture;
    texture.fromData(base(), 8, 8, 8, vk::Format::eR8Unorm, data.data(), vk::ImageUsageFlagBits::eSampled, true);
    ASSERT_EQ(texture.mipLevels, 4u);

    auto level1 = texture.download(1);
    ASSERT_EQ(level1.size(), size_t(4 * 4 * 4));
    for (auto v : level1) ASSERT_EQ(v, 77);

    auto level3 = texture.download(3);
    ASSERT_EQ(level3.size(), 1u);
    EXPECT_EQ(level3[0], 77);
}
