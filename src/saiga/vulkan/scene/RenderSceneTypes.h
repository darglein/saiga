/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/math.h"

#include <cstdint>
#include <vector>

namespace Saiga
{
namespace Vulkan
{
// The parameters of the RenderScene frame (docs/engine_design.md §10.1; S13: the frame shell; S17 adds the HDR frame)
struct RenderSceneParameters
{
    int msaaSamples = 4;
    vec4 clearColor = vec4(57, 57, 57, 255) / 255.0f;
    bool imgui      = true;
};

// The GPU times of one completed frame (timestamps), matched to the frame by its number
struct FrameTimes
{
    uint64_t frame = 0;                // RenderScene's frame number (1, 2, ...)
    double gpuMs   = 0;                // the whole frame
    std::vector<double> gpuSectionMs;  // per section of RenderScene::gpuSectionNames
};

}  // namespace Vulkan
}  // namespace Saiga
