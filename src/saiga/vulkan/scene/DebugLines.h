/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/math.h"
#include "saiga/vulkan/buffer/Buffer.h"
#include "saiga/vulkan/pipeline/Pipeline.h"

#include <array>
#include <vector>

namespace Saiga
{
namespace Vulkan
{
/**
 * Debug lines of RenderScene (S36, A8 of docs/stages/S36_lighting.plan.md; shader/vulkan/scene/lighting/
 * debug_lines.vert/.frag): line lists in world coordinates (double) with a display colour, made camera-relative on the
 * CPU and drawn at the end of the scene pass (GPU sub-section scene.debug), depth tested (reversed-Z, no depth write)
 * or on top. A general mechanism: the lighting's overlays use it (a1: the sun and moon directions), games can add
 * lines. The lines of a frame are added before renderFrame (or in its transfer callback) and cleared after the frame
 * drew them.
 */
class SAIGA_VULKAN_API DebugLines
{
   public:
    void init(VulkanBase& base, const RenderingFormats& formats);
    void destroy();
    ~DebugLines() { destroy(); }

    // A line of the next frame from a to b (world coordinates of the camera's space); color: sRGB display colour
    void addLine(const Vec3& a, const Vec3& b, const vec3& color, bool onTop = false);
    void clear() { lines.clear(); }
    size_t lineCount() const { return lines.size(); }

    // Inside the scene pass: the frame's lines relative to the camera; colorScale > 0: the scene writes linear values
    // (srgbToLinear(colour) * colorScale), 0: display values. Clears the lines; returns the number drawn
    size_t record(vk::CommandBuffer cmd, uint32_t frameIndex, const Vec3& camera, const mat4& viewProj,
                  float colorScale);
    uint64_t memoryBytes() const;

   private:
    struct Line
    {
        Vec3 a, b;
        vec3 color;
        bool onTop;
    };
    struct Vertex
    {
        vec3 position;
        vec4 color;
    };
    VulkanBase* base = nullptr;
    Pipeline depthTested, onTop;
    std::array<Buffer, 2> vertices;
    std::vector<Line> lines;
};

}  // namespace Vulkan
}  // namespace Saiga
