/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/scene/SkyAmbient.h"

#include "saiga/vulkan/scene/EnvironmentMap.h"
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

bool SkyAmbient::usesEnvironmentMap() const
{
    return !uniform && environmentMap && environmentMap->valid();
}

void SkyAmbient::update(vk::CommandBuffer, uint32_t frameIndex, const EnvironmentState& environment)
{
    stats = AmbientStatistics();
    if (usesEnvironmentMap())
    {
        stats.sh       = environmentMap->sh();
        stats.dominant = stats.sh[0];
        return;
    }
    // A uniform radiance L: the irradiance is pi L for every normal (band 0 only); else the floor
    stats.sh[0]      = uniform ? vec3(*uniform * kPi) : environment.minAmbient;
    stats.dominant   = stats.sh[0];
    float data[9][4] = {};
    for (int i = 0; i < 9; ++i)
        for (int c = 0; c < 3; ++c) data[i][c] = stats.sh[size_t(i)](c);
    std::memcpy(buffers[frameIndex].getMappedPointer(), data, sizeof(data));
}

AmbientBindings SkyAmbient::bindings(uint32_t frameIndex)
{
    AmbientBindings b;
    if (usesEnvironmentMap())
    {
        b.sh          = environmentMap->frontSh();
        b.shBytes     = EnvironmentMap::kShBytes;
        b.environment = environmentMap->frontCubeInfo();
        return b;
    }
    b.sh      = handle(buffers[frameIndex]);
    b.shBytes = kShBytes;
    return b;
}

}  // namespace Vulkan
}  // namespace Saiga
