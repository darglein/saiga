/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
// PipelineInfo::setState: the API independent pipeline state in Vulkan's structs. Needs no device.

#include "saiga/vulkan/pipeline/PipelineInfo.h"

#include "gtest/gtest.h"

#include <algorithm>

using namespace Saiga;
using namespace Saiga::Vulkan;

TEST(VulkanPipelineState, DefaultsTurnEverythingOff)
{
    PipelineInfo info;
    info.setState(PipelineState());
    EXPECT_FALSE(info.inputAssemblyState.primitiveRestartEnable);
    EXPECT_EQ(info.rasterizationState.polygonMode, vk::PolygonMode::eFill);
    EXPECT_EQ(info.rasterizationState.cullMode, vk::CullModeFlags(vk::CullModeFlagBits::eNone));
    EXPECT_EQ(info.rasterizationState.frontFace, vk::FrontFace::eCounterClockwise);
    EXPECT_FALSE(info.rasterizationState.depthBiasEnable);
    EXPECT_FALSE(info.depthStencilState.depthTestEnable);
    EXPECT_FALSE(info.depthStencilState.depthWriteEnable);
    EXPECT_FALSE(info.depthStencilState.stencilTestEnable);
    EXPECT_FALSE(info.blendAttachmentState.blendEnable);
    EXPECT_EQ(info.blendAttachmentState.colorWriteMask,
              vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG | vk::ColorComponentFlagBits::eB |
                  vk::ColorComponentFlagBits::eA);
    // Without the stencil test, the reference stays static.
    EXPECT_EQ(std::count(info.dynamicStateEnables.begin(), info.dynamicStateEnables.end(),
                         vk::DynamicState::eStencilReference),
              0);
}

TEST(VulkanPipelineState, EveryField)
{
    PipelineState s;
    s.primitiveRestart                = true;
    s.rasterization.polygonMode       = PolygonMode::Line;
    s.rasterization.cullMode          = CullMode::Front;
    s.rasterization.frontFace         = FrontFace::Clockwise;
    s.rasterization.depthBias         = true;
    s.rasterization.depthBiasConstant = 2.f;
    s.rasterization.depthBiasSlope    = 1.5f;
    s.depthStencil.depthTest          = true;
    s.depthStencil.depthWrite         = true;
    s.depthStencil.depthCompare       = CompareOp::GreaterOrEqual;
    s.depthStencil.stencilTest        = true;
    s.depthStencil.front.failOp       = StencilOp::Zero;
    s.depthStencil.front.passOp       = StencilOp::Replace;
    s.depthStencil.front.depthFailOp  = StencilOp::IncrementAndWrap;
    s.depthStencil.front.compareOp    = CompareOp::NotEqual;
    s.depthStencil.front.compareMask  = 0x0F;
    s.depthStencil.front.writeMask    = 0xF0;
    s.depthStencil.back.depthFailOp   = StencilOp::DecrementAndClamp;
    s.blend                           = BlendState::premultipliedAlpha();
    s.blend.alphaOp                   = BlendOp::Max;
    s.blend.colorWriteMask            = ColorComponentR | ColorComponentB;

    PipelineInfo info;
    info.setState(s);
    info.setState(s);  // the reference becomes dynamic only once

    EXPECT_TRUE(info.inputAssemblyState.primitiveRestartEnable);
    EXPECT_EQ(info.rasterizationState.polygonMode, vk::PolygonMode::eLine);
    EXPECT_EQ(info.rasterizationState.cullMode, vk::CullModeFlags(vk::CullModeFlagBits::eFront));
    EXPECT_EQ(info.rasterizationState.frontFace, vk::FrontFace::eClockwise);
    EXPECT_TRUE(info.rasterizationState.depthBiasEnable);
    EXPECT_EQ(info.rasterizationState.depthBiasConstantFactor, 2.f);
    EXPECT_EQ(info.rasterizationState.depthBiasSlopeFactor, 1.5f);

    EXPECT_TRUE(info.depthStencilState.depthTestEnable);
    EXPECT_TRUE(info.depthStencilState.depthWriteEnable);
    EXPECT_EQ(info.depthStencilState.depthCompareOp, vk::CompareOp::eGreaterOrEqual);
    EXPECT_TRUE(info.depthStencilState.stencilTestEnable);
    EXPECT_EQ(info.depthStencilState.front.failOp, vk::StencilOp::eZero);
    EXPECT_EQ(info.depthStencilState.front.passOp, vk::StencilOp::eReplace);
    EXPECT_EQ(info.depthStencilState.front.depthFailOp, vk::StencilOp::eIncrementAndWrap);
    EXPECT_EQ(info.depthStencilState.front.compareOp, vk::CompareOp::eNotEqual);
    EXPECT_EQ(info.depthStencilState.front.compareMask, 0x0Fu);
    EXPECT_EQ(info.depthStencilState.front.writeMask, 0xF0u);
    EXPECT_EQ(info.depthStencilState.back.depthFailOp, vk::StencilOp::eDecrementAndClamp);
    EXPECT_EQ(std::count(info.dynamicStateEnables.begin(), info.dynamicStateEnables.end(),
                         vk::DynamicState::eStencilReference),
              1);

    EXPECT_TRUE(info.blendAttachmentState.blendEnable);
    EXPECT_EQ(info.blendAttachmentState.srcColorBlendFactor, vk::BlendFactor::eOne);
    EXPECT_EQ(info.blendAttachmentState.dstColorBlendFactor, vk::BlendFactor::eOneMinusSrcAlpha);
    EXPECT_EQ(info.blendAttachmentState.colorBlendOp, vk::BlendOp::eAdd);
    EXPECT_EQ(info.blendAttachmentState.alphaBlendOp, vk::BlendOp::eMax);
    EXPECT_EQ(info.blendAttachmentState.colorWriteMask,
              vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eB);
}
