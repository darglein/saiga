/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/VulkanAsset.h"
#include "saiga/vulkan/renderModules/AssetRenderer.h"
#include "saiga/vulkan/renderModules/TextureDisplay.h"

#include "vulkan_test_context.h"

using namespace Saiga;

// The forward renderer with a HeadlessWindow: it renders into offscreen images, which are read back.
class VulkanHeadlessRenderer : public VulkanTest
{
};

TEST_F(VulkanHeadlessRenderer, ClearColor)
{
    auto img = RenderFrame();
    ASSERT_EQ(img.w, width);
    ASSERT_EQ(img.h, height);
    for (int y = 0; y < img.h; ++y)
        for (int x = 0; x < img.w; ++x) ASSERT_EQ(img(y, x), ClearColor()) << "at " << x << "," << y;
}

// More frames than frames in flight, so every offscreen image is reused.
TEST_F(VulkanHeadlessRenderer, ManyFrames)
{
    int rendered   = 0;
    app.onRender   = [&](vk::CommandBuffer) { rendered++; };
    int transfered = 0;
    app.onTransfer = [&](vk::CommandBuffer) { transfered++; };

    for (int i = 0; i < 5; ++i)
    {
        auto img = RenderFrame();
        ASSERT_EQ(img.w, width);
        EXPECT_EQ(img(0, 0), ClearColor());
    }
    window->renderFrames(4);

    EXPECT_EQ(rendered, 9);
    EXPECT_EQ(transfered, 9);
    EXPECT_EQ(app.updates, 9);
}

TEST_F(VulkanHeadlessRenderer, TextureDisplay)
{
    Vulkan::Texture2D texture;
    {
        TemplatedImage<ucvec4> img(4, 4);
        img.getImageView().set(ucvec4(255, 0, 0, 255));
        texture.fromImage(base(), img);
    }

    Vulkan::TextureDisplay display;
    display.init(base(), renderer->renderingFormats());
    auto set = display.createAndUpdateDescriptorSet(texture);

    // A red rectangle at [8,24) x [4,20)
    bool bound   = false;
    app.onRender = [&](vk::CommandBuffer cmd)
    {
        bound = display.bind(cmd);
        if (bound) display.renderTexture(cmd, set, vec2(8, 4), vec2(16, 16));
    };

    auto img = RenderFrame();
    ASSERT_TRUE(bound);
    ASSERT_EQ(img.w, width);

    ucvec4 red(255, 0, 0, 255);
    EXPECT_EQ(img(4, 8), red);
    EXPECT_EQ(img(12, 16), red);
    EXPECT_EQ(img(19, 23), red);

    EXPECT_EQ(img(3, 8), ClearColor());
    EXPECT_EQ(img(4, 7), ClearColor());
    EXPECT_EQ(img(20, 23), ClearColor());
    EXPECT_EQ(img(19, 24), ClearColor());
    EXPECT_EQ(img(40, 50), ClearColor());
}

TEST_F(VulkanHeadlessRenderer, ColoredAsset)
{
    // A triangle around the center in clip space (identity view and projection). Both windings, so it is visible
    // independent of the culling.
    Vulkan::VulkanVertexColoredAsset triangle;
    for (vec3 p : {vec3(-0.5f, -0.5f, 0.5f), vec3(0.5f, -0.5f, 0.5f), vec3(0, 0.5f, 0.5f)})
    {
        VertexNC v(p, vec3(0, 0, 1), vec3(0, 1, 0));
        triangle.addVertex(v);
    }
    triangle.addFace(0, 1, 2);
    triangle.addFace(0, 2, 1);
    triangle.init(base());

    Vulkan::AssetRenderer assetRenderer;
    assetRenderer.init(base(), renderer->renderingFormats());

    app.onTransfer = [&](vk::CommandBuffer cmd)
    { assetRenderer.updateUniformBuffers(cmd, mat4::Identity(), mat4::Identity()); };

    bool bound   = false;
    app.onRender = [&](vk::CommandBuffer cmd)
    {
        bound = assetRenderer.bind(cmd);
        if (bound)
        {
            assetRenderer.pushModel(cmd, mat4::Identity());
            triangle.render(cmd);
        }
    };

    auto img = RenderFrame();
    ASSERT_TRUE(bound);
    ASSERT_EQ(img.w, width);

    // The shaded color depends on the lighting, but green is the only channel of the vertex color.
    ucvec4 center = img(height / 2, width / 2);
    EXPECT_NE(center, ClearColor());
    EXPECT_GT(center(1), center(0));
    EXPECT_GT(center(1), center(2));

    EXPECT_EQ(img(0, 0), ClearColor());
    EXPECT_EQ(img(height - 1, 0), ClearColor());
    EXPECT_EQ(img(0, width - 1), ClearColor());
    EXPECT_EQ(img(height - 1, width - 1), ClearColor());
}
