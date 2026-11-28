/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <cstdint>

namespace Saiga
{
// The fixed-function state of a draw: rasterization, depth, stencil and blending, independent of the graphics API.
// The OpenGL module sets it with applyPipelineState (saiga/opengl/PipelineState.h), the Vulkan module fills a
// PipelineInfo from it (PipelineInfo::setState).
//
// Every field has a value, and the defaults turn everything off: no culling, no depth or stencil test, no depth bias,
// no blending, no scissor test, all color channels written. A pass states what it needs and nothing of an earlier pass
// carries over.
//
// The values that change between draws without a new pipeline in Vulkan are not part of it: the viewport, the scissor
// rectangle, the line width and the stencil reference value.

enum class CompareOp : uint8_t
{
    Never,
    Less,
    Equal,
    LessOrEqual,
    Greater,
    NotEqual,
    GreaterOrEqual,
    Always
};

enum class StencilOp : uint8_t
{
    Keep,
    Zero,
    Replace,
    IncrementAndClamp,
    DecrementAndClamp,
    Invert,
    IncrementAndWrap,
    DecrementAndWrap
};

enum class BlendFactor : uint8_t
{
    Zero,
    One,
    SrcColor,
    OneMinusSrcColor,
    DstColor,
    OneMinusDstColor,
    SrcAlpha,
    OneMinusSrcAlpha,
    DstAlpha,
    OneMinusDstAlpha
};

enum class BlendOp : uint8_t
{
    Add,
    Subtract,
    ReverseSubtract,
    Min,
    Max
};

enum class CullMode : uint8_t
{
    None,
    Front,
    Back,
    FrontAndBack
};

// The winding of the front faces as seen in the image, with row 0 of the framebuffer at the top (Vulkan) or at the
// bottom (OpenGL, see ClipSpace in saiga/opengl/pipeline.h).
enum class FrontFace : uint8_t
{
    CounterClockwise,
    Clockwise
};

enum class PolygonMode : uint8_t
{
    Fill,
    Line,
    Point
};

// Bits of BlendState::colorWriteMask.
enum ColorComponent : uint8_t
{
    ColorComponentR   = 1,
    ColorComponentG   = 2,
    ColorComponentB   = 4,
    ColorComponentA   = 8,
    ColorComponentAll = 15
};

struct RasterizationState
{
    PolygonMode polygonMode = PolygonMode::Fill;
    CullMode cullMode       = CullMode::None;
    FrontFace frontFace     = FrontFace::CounterClockwise;

    // Offsets the depth of polygons by depthBiasSlope * (their depth slope) + depthBiasConstant * (the smallest
    // resolvable depth difference), as glPolygonOffset and Vulkan's depth bias do.
    bool depthBias          = false;
    float depthBiasConstant = 0;
    float depthBiasSlope    = 0;

    // The rectangle is dynamic state. Vulkan always clips to it, so a disabled test there means a rectangle that covers
    // the viewport.
    bool scissorTest = false;

    bool operator==(const RasterizationState& o) const
    {
        return polygonMode == o.polygonMode && cullMode == o.cullMode && frontFace == o.frontFace &&
               depthBias == o.depthBias && depthBiasConstant == o.depthBiasConstant &&
               depthBiasSlope == o.depthBiasSlope && scissorTest == o.scissorTest;
    }
    bool operator!=(const RasterizationState& o) const { return !(*this == o); }
};

struct StencilFaceState
{
    StencilOp failOp      = StencilOp::Keep;
    StencilOp passOp      = StencilOp::Keep;
    StencilOp depthFailOp = StencilOp::Keep;
    CompareOp compareOp   = CompareOp::Always;
    uint32_t compareMask  = 0xFF;
    uint32_t writeMask    = 0xFF;

    bool operator==(const StencilFaceState& o) const
    {
        return failOp == o.failOp && passOp == o.passOp && depthFailOp == o.depthFailOp && compareOp == o.compareOp &&
               compareMask == o.compareMask && writeMask == o.writeMask;
    }
    bool operator!=(const StencilFaceState& o) const { return !(*this == o); }
};

struct DepthStencilState
{
    bool depthTest = false;
    // Depth writes only happen with the depth test enabled, as in Vulkan; OpenGL writes no depth without it either.
    bool depthWrite        = false;
    CompareOp depthCompare = CompareOp::Less;

    bool stencilTest = false;
    StencilFaceState front;
    StencilFaceState back;

    // The same stencil state for front and back faces.
    void setStencil(const StencilFaceState& state) { front = back = state; }

    bool operator==(const DepthStencilState& o) const
    {
        return depthTest == o.depthTest && depthWrite == o.depthWrite && depthCompare == o.depthCompare &&
               stencilTest == o.stencilTest && front == o.front && back == o.back;
    }
    bool operator!=(const DepthStencilState& o) const { return !(*this == o); }
};

// The blending of all color attachments.
struct BlendState
{
    bool enable          = false;
    BlendFactor srcColor = BlendFactor::One;
    BlendFactor dstColor = BlendFactor::Zero;
    BlendOp colorOp      = BlendOp::Add;
    BlendFactor srcAlpha = BlendFactor::One;
    BlendFactor dstAlpha = BlendFactor::Zero;
    BlendOp alphaOp      = BlendOp::Add;

    uint8_t colorWriteMask = ColorComponentAll;

    // color * alpha + destination * (1 - alpha); the alpha channel stays correct for blending it again later.
    static BlendState alpha()
    {
        BlendState b;
        b.enable   = true;
        b.srcColor = BlendFactor::SrcAlpha;
        b.dstColor = BlendFactor::OneMinusSrcAlpha;
        b.srcAlpha = BlendFactor::One;
        b.dstAlpha = BlendFactor::OneMinusSrcAlpha;
        return b;
    }

    // For premultiplied colors: color + destination * (1 - alpha).
    static BlendState premultipliedAlpha()
    {
        BlendState b;
        b.enable   = true;
        b.srcColor = BlendFactor::One;
        b.dstColor = BlendFactor::OneMinusSrcAlpha;
        b.srcAlpha = BlendFactor::One;
        b.dstAlpha = BlendFactor::OneMinusSrcAlpha;
        return b;
    }

    // color + destination, in all channels.
    static BlendState additive()
    {
        BlendState b;
        b.enable   = true;
        b.srcColor = BlendFactor::One;
        b.dstColor = BlendFactor::One;
        b.srcAlpha = BlendFactor::One;
        b.dstAlpha = BlendFactor::One;
        return b;
    }

    bool operator==(const BlendState& o) const
    {
        return enable == o.enable && srcColor == o.srcColor && dstColor == o.dstColor && colorOp == o.colorOp &&
               srcAlpha == o.srcAlpha && dstAlpha == o.dstAlpha && alphaOp == o.alphaOp &&
               colorWriteMask == o.colorWriteMask;
    }
    bool operator!=(const BlendState& o) const { return !(*this == o); }
};

struct PipelineState
{
    // Primitive restart with the largest index of the index type.
    bool primitiveRestart = false;

    RasterizationState rasterization;
    DepthStencilState depthStencil;
    BlendState blend;

    // An opaque 3D pass: depth test and depth writes.
    static PipelineState opaque(CompareOp depthCompare = CompareOp::Less)
    {
        PipelineState s;
        s.depthStencil.depthTest    = true;
        s.depthStencil.depthWrite   = true;
        s.depthStencil.depthCompare = depthCompare;
        return s;
    }

    bool operator==(const PipelineState& o) const
    {
        return primitiveRestart == o.primitiveRestart && rasterization == o.rasterization &&
               depthStencil == o.depthStencil && blend == o.blend;
    }
    bool operator!=(const PipelineState& o) const { return !(*this == o); }
};

}  // namespace Saiga
