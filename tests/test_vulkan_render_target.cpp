/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/pipeline/Pipeline.h"
#include "saiga/vulkan/renderModules/TextureDisplay.h"
#include "saiga/vulkan/texture/RenderTarget.h"

#include <cstring>

#include "vulkan_test_context.h"

using namespace Saiga;

class VulkanRenderTarget : public VulkanTest
{
   protected:
    // A triangle from the vertex index, without vertex buffers: the lower left half of the target (below the
    // diagonal from the top left to the bottom right corner).
    static constexpr const char* triangle_vert = R"(#version 450
void main()
{
    vec2 positions[3] = vec2[](vec2(-1, -1), vec2(1, 1), vec2(-1, 1));
    gl_Position = vec4(positions[gl_VertexIndex], 0.5, 1);
}
)";

    static constexpr const char* white_frag = R"(#version 450
layout(location = 0) out vec4 outColor;
void main()
{
    outColor = vec4(1);
}
)";

    static std::vector<ucvec4> Pixels(Vulkan::Texture2D& texture)
    {
        auto data = texture.download();
        std::vector<ucvec4> result(data.size() / 4);
        std::memcpy(result.data(), data.data(), result.size() * 4);
        return result;
    }

    // Draws the white triangle into a black target and returns the red channel of the pixels.
    static std::vector<int> RenderTriangle(vk::SampleCountFlagBits samples)
    {
        Vulkan::RenderTargetParameters params;
        params.samples = samples;
        Vulkan::RenderTarget target;
        target.create(base(), 32, 32, params);

        Vulkan::Pipeline pipeline;
        pipeline.init(base(), 0);
        EXPECT_TRUE(pipeline.shaderPipeline.loadGLSLStrings(
            base().device,
            {{vk::ShaderStageFlagBits::eVertex, triangle_vert}, {vk::ShaderStageFlagBits::eFragment, white_frag}}));
        Vulkan::PipelineInfo info;
        info.rasterizationState.cullMode           = vk::CullModeFlagBits::eNone;
        info.multisampleState.rasterizationSamples = samples;
        pipeline.create(target.renderingFormats(), info);

        Submit(
            [&](vk::CommandBuffer cmd)
            {
                target.begin(cmd);
                if (pipeline.bind(cmd)) cmd.draw(3, 1, 0, 0);
                target.end(cmd);
            });

        std::vector<int> result;
        for (auto p : Pixels(target.colorTexture())) result.push_back(p(0));
        return result;
    }
};

TEST_F(VulkanRenderTarget, Clear)
{
    Vulkan::RenderTarget target;
    target.create(base(), 20, 10);
    EXPECT_EQ(target.width(), 20u);
    EXPECT_EQ(target.height(), 10u);
    EXPECT_EQ(target.colorCount(), 1u);
    ASSERT_TRUE(target.hasDepth());

    auto formats = target.renderingFormats();
    ASSERT_EQ(formats.color.size(), 1u);
    EXPECT_EQ(formats.color[0], vk::Format::eR8G8B8A8Unorm);
    EXPECT_EQ(formats.depth, vk::Format::eD32Sfloat);

    Submit(
        [&](vk::CommandBuffer cmd)
        {
            target.begin(cmd, true, vec4(1, 0, 0.5f, 1), 0.25f);
            target.end(cmd);
        });

    // Readable by shaders after end()
    EXPECT_EQ(target.colorTexture().memoryLocation->data.layout, vk::ImageLayout::eShaderReadOnlyOptimal);

    auto pixels = Pixels(target.colorTexture());
    ASSERT_EQ(pixels.size(), size_t(20 * 10));
    for (auto p : pixels)
    {
        ASSERT_EQ(p(0), 255);
        ASSERT_EQ(p(1), 0);
        ASSERT_NEAR(p(2), 128, 1);
        ASSERT_EQ(p(3), 255);
    }

    auto depth_data = target.depthTexture().download();
    ASSERT_EQ(depth_data.size(), size_t(20 * 10) * sizeof(float));
    std::vector<float> depth(20 * 10);
    std::memcpy(depth.data(), depth_data.data(), depth_data.size());
    for (auto d : depth) ASSERT_EQ(d, 0.25f);
}

// clear = false keeps the content of the previous pass.
TEST_F(VulkanRenderTarget, Load)
{
    Vulkan::RenderTarget target;
    target.create(base(), 8, 8);

    Submit(
        [&](vk::CommandBuffer cmd)
        {
            target.begin(cmd, true, vec4(0, 1, 0, 1));
            target.end(cmd);
            target.begin(cmd, false);
            target.end(cmd);
        });

    for (auto p : Pixels(target.colorTexture())) ASSERT_EQ(p, ucvec4(0, 255, 0, 255));
}

TEST_F(VulkanRenderTarget, DrawAndSample)
{
    Vulkan::RenderTarget target;
    target.create(base(), 16, 16);

    Vulkan::Texture2D green;
    {
        TemplatedImage<ucvec4> img(2, 2);
        img.getImageView().set(ucvec4(0, 255, 0, 255));
        green.fromImage(base(), img);
    }

    // First pass: a green square in the top left quarter of the target.
    Vulkan::TextureDisplay toTarget;
    toTarget.init(base(), target.renderingFormats());
    auto greenSet = toTarget.createAndUpdateDescriptorSet(green);

    // Second pass: the target as a texture in the frame.
    Vulkan::TextureDisplay toFrame;
    toFrame.init(base(), renderer->renderingFormats());
    auto targetSet = toFrame.createDescriptorSet();

    app.onTransfer = [&](vk::CommandBuffer cmd)
    {
        target.begin(cmd, true, vec4(0, 0, 1, 1));
        if (toTarget.bind(cmd)) toTarget.renderTexture(cmd, greenSet, vec2(0, 0), vec2(8, 8));
        target.end(cmd);
    };
    app.onRender = [&](vk::CommandBuffer cmd)
    {
        if (toFrame.bind(cmd)) toFrame.renderTexture(cmd, targetSet, vec2(0, 0), vec2(32, 32));
    };
    // The descriptor needs the final layout of the target, which is only tracked while recording: render it once.
    Submit(
        [&](vk::CommandBuffer cmd)
        {
            target.begin(cmd);
            target.end(cmd);
        });
    targetSet.assign(0, &target.colorTexture());

    auto frame = RenderFrame();
    ASSERT_EQ(frame.w, width);

    // The first pass: green in the top left quarter of the target, the clear color elsewhere. Only the color is
    // compared: TextureDisplay blends, and the alpha blend factors of the default PipelineInfo do not keep alpha.
    auto pixels = Pixels(target.colorTexture());
    ASSERT_EQ(pixels.size(), size_t(16 * 16));
    EXPECT_EQ(pixels[4 * 16 + 4](0), 0);
    EXPECT_EQ(pixels[4 * 16 + 4](1), 255);
    EXPECT_EQ(pixels[4 * 16 + 4](2), 0);
    EXPECT_EQ(pixels[4 * 16 + 12], ucvec4(0, 0, 255, 255));
    EXPECT_EQ(pixels[12 * 16 + 4], ucvec4(0, 0, 255, 255));

    // The second pass: the 16x16 target magnified to 32x32 in the frame.
    ucvec4 blue(0, 0, 255, 255);
    EXPECT_EQ(frame(8, 8), blue);
    EXPECT_EQ(frame(8, 24), blue);
    EXPECT_EQ(frame(24, 24), blue);
    EXPECT_EQ(frame(8, 40), ClearColor());
    EXPECT_EQ(frame(40, 8), ClearColor());
}

TEST_F(VulkanRenderTarget, MultipleRenderTargets)
{
    Vulkan::RenderTargetParameters params;
    params.colorFormats = {vk::Format::eR8G8B8A8Unorm, vk::Format::eR32Uint};
    params.depthFormat  = vk::Format::eUndefined;

    Vulkan::RenderTarget target;
    target.create(base(), 8, 8, params);
    EXPECT_FALSE(target.hasDepth());
    ASSERT_EQ(target.colorCount(), 2u);

    static const char* frag = R"(#version 450
layout(location = 0) out vec4 outColor;
layout(location = 1) out uint outId;
void main()
{
    outColor = vec4(0, 1, 0, 1);
    outId    = 42;
}
)";

    Vulkan::Pipeline pipeline;
    pipeline.init(base(), 0);
    ASSERT_TRUE(pipeline.shaderPipeline.loadGLSLStrings(
        base().device,
        {{vk::ShaderStageFlagBits::eVertex, triangle_vert}, {vk::ShaderStageFlagBits::eFragment, frag}}));
    Vulkan::PipelineInfo info;
    info.rasterizationState.cullMode        = vk::CullModeFlagBits::eNone;
    info.depthStencilState.depthTestEnable  = VK_FALSE;
    info.depthStencilState.depthWriteEnable = VK_FALSE;
    pipeline.create(target.renderingFormats(), info);

    bool bound = false;
    Submit(
        [&](vk::CommandBuffer cmd)
        {
            // Integer attachments are cleared with the integer values of the clear color.
            target.begin(cmd, true, vec4(7, 0, 0, 1));
            bound = pipeline.bind(cmd);
            if (bound) cmd.draw(3, 1, 0, 0);
            target.end(cmd);
        });
    ASSERT_TRUE(bound);

    auto colors  = Pixels(target.colorTexture(0));
    auto id_data = target.colorTexture(1).download();
    ASSERT_EQ(id_data.size(), size_t(8 * 8) * sizeof(uint32_t));
    std::vector<uint32_t> ids(8 * 8);
    std::memcpy(ids.data(), id_data.data(), id_data.size());

    // Row 7, column 0 is inside the triangle (bottom left), row 0, column 7 outside (top right).
    EXPECT_EQ(colors[7 * 8 + 0], ucvec4(0, 255, 0, 255));
    EXPECT_EQ(ids[7 * 8 + 0], 42u);
    EXPECT_EQ(colors[0 * 8 + 7], ucvec4(255, 0, 0, 255));
    EXPECT_EQ(ids[0 * 8 + 7], 7u);
}

// With multisampling the pixels on the diagonal edge are resolved to intermediate values.
TEST_F(VulkanRenderTarget, Multisampling)
{
    auto count_intermediate = [](const std::vector<int>& pixels)
    {
        int n = 0;
        for (int p : pixels)
            if (p != 0 && p != 255) n++;
        return n;
    };

    auto aliased = RenderTriangle(vk::SampleCountFlagBits::e1);
    ASSERT_EQ(aliased.size(), size_t(32 * 32));
    EXPECT_EQ(count_intermediate(aliased), 0);
    EXPECT_EQ(aliased[31 * 32 + 0], 255);
    EXPECT_EQ(aliased[0 * 32 + 31], 0);

    auto limits = base().physicalDevice.getProperties().limits;
    auto counts = limits.framebufferColorSampleCounts & limits.framebufferDepthSampleCounts;
    if (!(counts & vk::SampleCountFlagBits::e4)) GTEST_SKIP() << "4x multisampling is not supported";

    auto smooth = RenderTriangle(vk::SampleCountFlagBits::e4);
    ASSERT_EQ(smooth.size(), size_t(32 * 32));
    EXPECT_GT(count_intermediate(smooth), 0);
    EXPECT_EQ(smooth[31 * 32 + 0], 255);
    EXPECT_EQ(smooth[0 * 32 + 31], 0);
}
