/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/scene/SkyAmbient.h"

#include "saiga/vulkan/scene/RenderSceneTypes.h"

#include <algorithm>
#include <cstring>

namespace Saiga
{
namespace Vulkan
{
namespace
{
constexpr float kPi = 3.14159265358979f;

vk::Buffer handle(Buffer& b)
{
    Memory::SafeAccessor acc(*b.memoryLocation());
    return b.memoryLocation()->data;
}
}  // namespace

void SkyAmbient::init(VulkanBase& b)
{
    base = &b;
    for (Buffer& buffer : buffers)
    {
        buffer.createBuffer(b, kShBytes, vk::BufferUsageFlagBits::eStorageBuffer,
                            vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        std::memset(buffer.getMappedPointer(), 0, kShBytes);
    }
}

void SkyAmbient::destroy()
{
    if (!base) return;
    for (Buffer& buffer : buffers) buffer.destroy();
    base = nullptr;
}

vec3 SkyAmbient::skyRadiance(const EnvironmentState& e)
{
    // The bodies' illuminance on a horizontal plane (above the horizon), a blue sky of 8% of it spread over the
    // hemisphere (radiance = E / pi)
    const float sunE  = e.sunIlluminance * std::max(e.sunDir(1), 0.0f);
    const float moonE = e.moonIlluminance * std::max(e.moonDir(1), 0.0f);
    return vec3(0.55f, 0.7f, 1.0f) * (0.08f * (sunE + moonE) / kPi);
}

void SkyAmbient::update(vk::CommandBuffer, uint32_t frameIndex, const EnvironmentState& environment)
{
    // A uniform radiance L: the irradiance is pi L for every normal (band 0 only)
    const vec3 L     = uniform ? *uniform : skyRadiance(environment);
    stats            = AmbientStatistics();
    stats.sh[0]      = L * kPi;
    stats.dominant   = stats.sh[0];
    float data[9][4] = {};
    for (int i = 0; i < 9; ++i)
        for (int c = 0; c < 3; ++c) data[i][c] = stats.sh[size_t(i)](c);
    std::memcpy(buffers[frameIndex].getMappedPointer(), data, sizeof(data));
}

AmbientBindings SkyAmbient::bindings(uint32_t frameIndex)
{
    AmbientBindings b;
    b.sh      = handle(buffers[frameIndex]);
    b.shBytes = kShBytes;
    return b;
}

}  // namespace Vulkan
}  // namespace Saiga
