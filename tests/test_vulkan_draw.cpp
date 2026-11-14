/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/buffer/IndexBuffer.h"
#include "saiga/vulkan/buffer/IndirectBuffer.h"
#include "saiga/vulkan/buffer/VertexBuffer.h"
#include "saiga/vulkan/pipeline/Pipeline.h"
#include "saiga/vulkan/texture/RenderTarget.h"

#include <cstddef>
#include <cstring>

#include "vulkan_test_context.h"

using namespace Saiga;

// Instanced and indirect draws: a small quad is drawn once per instance, with the offset and color of the instance.
class VulkanDraw : public VulkanTest
{
   protected:
    struct Vertex
    {
        float x, y;
    };

    struct Instance
    {
        float offset_x, offset_y;
        float r, g, b, a;
    };

    static constexpr int size = 32;

    static constexpr const char* vert = R"(#version 450
layout(location = 0) in vec2 inPosition;
layout(location = 1) in vec2 inOffset;
layout(location = 2) in vec4 inColor;
layout(location = 0) out vec4 outColor;
void main()
{
    outColor    = inColor;
    gl_Position = vec4(inPosition + inOffset, 0.5, 1);
}
)";

    static constexpr const char* frag = R"(#version 450
layout(location = 0) in vec4 inColor;
layout(location = 0) out vec4 outColor;
void main()
{
    outColor = inColor;
}
)";

    // One instance per quadrant: top left, top right, bottom left, bottom right (y = -1 is the top in Vulkan).
    static std::vector<Instance> Instances()
    {
        return {
            {-0.5f, -0.5f, 1, 0, 0, 1},
            {0.5f, -0.5f, 0, 1, 0, 1},
            {-0.5f, 0.5f, 0, 0, 1, 1},
            {0.5f, 0.5f, 1, 1, 1, 1},
        };
    }

    static ucvec4 Black() { return ucvec4(0, 0, 0, 255); }
    static std::vector<ucvec4> QuadrantColors()
    {
        return {ucvec4(255, 0, 0, 255), ucvec4(0, 255, 0, 255), ucvec4(0, 0, 255, 255), ucvec4(255, 255, 255, 255)};
    }

    void SetUp() override
    {
        VulkanTest::SetUp();
        if (IsSkipped()) return;

        Vulkan::RenderTargetParameters params;
        params.depthFormat = vk::Format::eUndefined;
        target.create(base(), size, size, params);

        pipeline.init(base(), 0);
        ASSERT_TRUE(pipeline.shaderPipeline.loadGLSLStrings(
            base().device, {{vk::ShaderStageFlagBits::eVertex, vert}, {vk::ShaderStageFlagBits::eFragment, frag}}));

        Vulkan::PipelineInfo info;
        info.rasterizationState.cullMode        = vk::CullModeFlagBits::eNone;
        info.depthStencilState.depthTestEnable  = VK_FALSE;
        info.depthStencilState.depthWriteEnable = VK_FALSE;
        info.addVertexBinding(0, sizeof(Vertex), vk::VertexInputRate::eVertex, {{0, 0, vk::Format::eR32G32Sfloat, 0}});
        info.addVertexBinding(1, sizeof(Instance), vk::VertexInputRate::eInstance,
                              {{1, 1, vk::Format::eR32G32Sfloat, offsetof(Instance, offset_x)},
                               {2, 1, vk::Format::eR32G32B32A32Sfloat, offsetof(Instance, r)}});
        pipeline.create(target.renderingFormats(), info);

        instances.initDeviceLocal(base(), Instances());
    }

    void TearDown() override
    {
        // Before the device of the test suite is destroyed.
        quad.destroy();
        quadIndexed.destroy();
        indices.destroy();
        instances.destroy();
        pipeline.destroy();
        target.destroy();
        VulkanTest::TearDown();
    }

    // The quad [-0.25, 0.25]^2 as two triangles
    void CreateQuad()
    {
        float s = 0.25f;
        quad.initDeviceLocal(base(), {{-s, -s}, {s, -s}, {s, s}, {-s, -s}, {s, s}, {-s, s}});
    }

    void CreateIndexedQuad()
    {
        float s = 0.25f;
        quadIndexed.initDeviceLocal(base(), {{-s, -s}, {s, -s}, {s, s}, {-s, s}});
        indices.init(base(), std::vector<uint32_t>{0, 1, 2, 0, 2, 3}, vk::MemoryPropertyFlagBits::eDeviceLocal);
    }

    // Renders into the target and returns the colors at the centers of the 4 quadrants and at the center.
    std::vector<ucvec4> Render(const std::function<void(vk::CommandBuffer)>& draw)
    {
        bool bound = false;
        Submit(
            [&](vk::CommandBuffer cmd)
            {
                target.begin(cmd);
                bound = pipeline.bind(cmd);
                if (bound) draw(cmd);
                target.end(cmd);
            });
        EXPECT_TRUE(bound);

        auto data = target.colorTexture().download();
        std::vector<ucvec4> pixels(size * size);
        std::memcpy(pixels.data(), data.data(), data.size());
        auto at = [&](int x, int y) { return pixels[y * size + x]; };
        int a = size / 4, b = 3 * size / 4;
        return {at(a, a), at(b, a), at(a, b), at(b, b), at(size / 2, size / 2)};
    }

    Vulkan::RenderTarget target;
    Vulkan::Pipeline pipeline;
    Vulkan::VertexBuffer<Vertex> quad, quadIndexed;
    Vulkan::IndexBuffer<uint32_t> indices;
    Vulkan::VertexBuffer<Instance> instances;
};

TEST_F(VulkanDraw, Instanced)
{
    CreateQuad();
    auto result = Render(
        [&](vk::CommandBuffer cmd)
        {
            quad.bind(cmd, 0);
            instances.bind(cmd, 1);
            quad.drawInstanced(cmd, 4);
        });

    auto colors = QuadrantColors();
    for (int i = 0; i < 4; ++i) EXPECT_EQ(result[i], colors[i]) << "quadrant " << i;
    EXPECT_EQ(result[4], Black());
}

TEST_F(VulkanDraw, InstancedIndexed)
{
    CreateIndexedQuad();
    // Only the instances 1 and 2
    auto result = Render(
        [&](vk::CommandBuffer cmd)
        {
            quadIndexed.bind(cmd, 0);
            instances.bind(cmd, 1);
            indices.bind(cmd);
            indices.drawInstanced(cmd, 2, 1);
        });

    auto colors = QuadrantColors();
    EXPECT_EQ(result[0], Black());
    EXPECT_EQ(result[1], colors[1]);
    EXPECT_EQ(result[2], colors[2]);
    EXPECT_EQ(result[3], Black());
}

TEST_F(VulkanDraw, Indirect)
{
    if (!base().enabledFeatures.drawIndirectFirstInstance) GTEST_SKIP() << "No drawIndirectFirstInstance";
    CreateQuad();

    // Two draws: the instances 0 and 3.
    Vulkan::IndirectBuffer<vk::DrawIndirectCommand> indirect;
    indirect.init(base(), 4);
    indirect.setCommands({vk::DrawIndirectCommand(6, 1, 0, 0), vk::DrawIndirectCommand(6, 1, 0, 3)});
    EXPECT_EQ(indirect.drawCount, 2u);

    auto colors = QuadrantColors();
    auto result = Render(
        [&](vk::CommandBuffer cmd)
        {
            quad.bind(cmd, 0);
            instances.bind(cmd, 1);
            indirect.draw(cmd);
        });
    EXPECT_EQ(result[0], colors[0]);
    EXPECT_EQ(result[1], Black());
    EXPECT_EQ(result[2], Black());
    EXPECT_EQ(result[3], colors[3]);

    // Only the second command
    result = Render(
        [&](vk::CommandBuffer cmd)
        {
            quad.bind(cmd, 0);
            instances.bind(cmd, 1);
            indirect.draw(cmd, 1, 1);
        });
    EXPECT_EQ(result[0], Black());
    EXPECT_EQ(result[3], colors[3]);
}

TEST_F(VulkanDraw, IndirectIndexed)
{
    if (!base().enabledFeatures.drawIndirectFirstInstance) GTEST_SKIP() << "No drawIndirectFirstInstance";
    CreateIndexedQuad();

    // One command with two instances (1, 2), one with a single instance (3).
    Vulkan::IndirectBuffer<> indirect;
    indirect.init(base(),
                  {vk::DrawIndexedIndirectCommand(6, 2, 0, 0, 1), vk::DrawIndexedIndirectCommand(6, 1, 0, 0, 3)});

    auto colors = QuadrantColors();
    auto result = Render(
        [&](vk::CommandBuffer cmd)
        {
            quadIndexed.bind(cmd, 0);
            instances.bind(cmd, 1);
            indices.bind(cmd);
            indirect.draw(cmd);
        });
    EXPECT_EQ(result[0], Black());
    EXPECT_EQ(result[1], colors[1]);
    EXPECT_EQ(result[2], colors[2]);
    EXPECT_EQ(result[3], colors[3]);
}
