/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/renderModules/TextureDisplay.h"
#include "saiga/vulkan/texture/Sampler.h"
#include "saiga/vulkan/texture/Texture.h"

#include "vulkan_test_context.h"

using namespace Saiga;

class VulkanSampler : public VulkanTest
{
   protected:
    // Renders a 2x1 texture (black | white) over the whole frame with the sampler and returns the red channel of
    // the middle row.
    static std::vector<int> RenderEdge(const Vulkan::Sampler* sampler)
    {
        TemplatedImage<ucvec4> img(1, 2);
        img(0, 0) = ucvec4(0, 0, 0, 255);
        img(0, 1) = ucvec4(255, 255, 255, 255);

        Vulkan::Texture2D texture;
        texture.fromImage(base(), img);

        Vulkan::TextureDisplay display;
        display.init(base(), renderer->renderingFormats());
        auto set = display.createDescriptorSet();
        if (sampler)
            set.assign(0, &texture, *sampler);
        else
            set.assign(0, &texture);

        app.onRender = [&](vk::CommandBuffer cmd)
        {
            if (display.bind(cmd)) display.renderTexture(cmd, set, vec2(0, 0), vec2(width, height));
        };
        auto frame   = RenderFrame();
        app.onRender = nullptr;
        renderer->waitIdle();

        std::vector<int> row;
        for (int x = 0; x < frame.w; ++x) row.push_back(frame(frame.h / 2, x)(0));
        return row;
    }
};

TEST_F(VulkanSampler, CreateAndMove)
{
    Vulkan::Sampler a;
    EXPECT_FALSE(bool(a));
    a.create(base());
    EXPECT_TRUE(bool(a));
    vk::Sampler handle = a;

    Vulkan::Sampler b = std::move(a);
    EXPECT_FALSE(bool(a));
    EXPECT_EQ(vk::Sampler(b), handle);

    b.destroy();
    EXPECT_FALSE(bool(b));
}

TEST_F(VulkanSampler, Parameters)
{
    Vulkan::SamplerParameters params;
    params.setFilter(vk::Filter::eNearest);
    params.setAddressMode(vk::SamplerAddressMode::eClampToEdge);
    // Clamped to the device limit, or ignored without the feature.
    params.maxAnisotropy = 1000.0f;
    params.compareOp     = vk::CompareOp::eLess;

    Vulkan::Sampler sampler;
    sampler.create(base(), params);
    EXPECT_TRUE(bool(sampler));
}

// A sampler that is assigned with the texture replaces the texture's default (linear) sampler.
TEST_F(VulkanSampler, NearestAndLinear)
{
    Vulkan::SamplerParameters params;
    params.setFilter(vk::Filter::eNearest);
    params.setAddressMode(vk::SamplerAddressMode::eClampToEdge);
    Vulkan::Sampler nearest;
    nearest.create(base(), params);

    auto row = RenderEdge(&nearest);
    ASSERT_EQ((int)row.size(), width);
    // A hard edge in the middle
    for (int x = 0; x < width / 2; ++x) ASSERT_EQ(row[x], 0) << "at " << x;
    for (int x = width / 2; x < width; ++x) ASSERT_EQ(row[x], 255) << "at " << x;

    params.setFilter(vk::Filter::eLinear);
    Vulkan::Sampler linear;
    linear.create(base(), params);

    row = RenderEdge(&linear);
    // Clamped at the borders, a gradient in between
    EXPECT_EQ(row[0], 0);
    EXPECT_EQ(row[width - 1], 255);
    EXPECT_GT(row[width / 2], 100);
    EXPECT_LT(row[width / 2], 156);
    for (int x = 1; x < width; ++x) ASSERT_GE(row[x], row[x - 1]) << "at " << x;

    // The default sampler of a texture is linear with repeat: the left border blends with the right texel.
    row = RenderEdge(nullptr);
    EXPECT_GT(row[0], 100);
    EXPECT_LT(row[0], 156);
    EXPECT_LT(row[width / 4], 10);
}
