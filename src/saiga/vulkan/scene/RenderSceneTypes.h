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
// The tone map of RenderScene's post pass (docs/engine_design.md §10.5). Identity is the clay mode: the scene shaders
// write display values, which are shown as they are (no exposure). Aces: the scene shaders write linear radiance
// (pre-exposed), tone mapped with the ACES fit of Narkowicz and the sRGB curve.
enum class ToneMap : uint8_t
{
    Identity,
    Aces
};

// The parameters of the RenderScene frame (docs/engine_design.md §10.1, §17.9; S13: the frame shell; S17: the HDR
// frame). Tone map and exposure can also be changed at run time (RenderScene::setToneMap, setExposure).
struct RenderSceneParameters
{
    // MSAA samples of the scene pass, clamped to the largest supported count not above it (RenderScene::formats())
    int msaaSamples = 4;
    // A display value; converted to linear (and pre-exposed) when the tone map is not Identity
    vec4 clearColor = vec4(57, 57, 57, 255) / 255.0f;
    bool imgui      = true;
    ToneMap toneMap = ToneMap::Identity;
    // Fixed until auto exposure (S37); ignored (1) with Identity
    float exposure = 1;
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
