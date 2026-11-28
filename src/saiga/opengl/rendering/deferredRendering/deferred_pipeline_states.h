/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/rendering/PipelineState.h"

namespace Saiga
{
// The passes the deferred and the uber deferred renderer have in common.

// The geometry pass into the G-buffer: opaque with back faces culled. With maskUsedPixels it writes 0 into the
// stencil buffer (cleared to 0xFF) where it draws, so the lighting can skip the empty pixels.
inline PipelineState GBufferPipelineState(bool maskUsedPixels, bool offsetGeometry, float offsetFactor,
                                          float offsetUnits, bool wireframe)
{
    PipelineState s          = PipelineState::opaque();
    s.rasterization.cullMode = CullMode::Back;
    if (maskUsedPixels)
    {
        s.depthStencil.stencilTest = true;
        StencilFaceState mark;
        mark.passOp = StencilOp::Replace;
        s.depthStencil.setStencil(mark);
    }
    if (offsetGeometry)
    {
        s.rasterization.depthBias         = true;
        s.rasterization.depthBiasSlope    = offsetFactor;
        s.rasterization.depthBiasConstant = offsetUnits;
    }
    if (wireframe) s.rasterization.polygonMode = PolygonMode::Line;
    return s;
}

// Copies the G-buffer's depth into the bound framebuffer with a fullscreen quad: depth only, every pixel.
inline PipelineState DepthCopyPipelineState()
{
    PipelineState s        = PipelineState::opaque(CompareOp::Always);
    s.blend.colorWriteMask = 0;
    return s;
}

// The forward pass of the application after the lighting: opaque with back faces culled.
inline PipelineState ForwardPipelineState()
{
    PipelineState s          = PipelineState::opaque();
    s.rasterization.cullMode = CullMode::Back;
    return s;
}

}  // namespace Saiga
