/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/scene/RenderSceneTypes.h"

#include <algorithm>
#include <cmath>

namespace Saiga
{
namespace Vulkan
{
uint32_t packSnorm2x16(float a, float b)
{
    auto pack = [](float v) -> uint32_t
    {
        const float c = std::round(std::clamp(v, -1.0f, 1.0f) * 32767.0f);
        return uint32_t(uint16_t(int16_t(c)));
    };
    return pack(a) | (pack(b) << 16);
}

vec2 unpackSnorm2x16(uint32_t v)
{
    auto unpack = [](uint32_t u) { return std::clamp(float(int16_t(uint16_t(u))) / 32767.0f, -1.0f, 1.0f); };
    return vec2(unpack(v & 0xFFFFu), unpack(v >> 16));
}

uint32_t packNormalOct(const vec3& n)
{
    const float l1 = std::abs(n(0)) + std::abs(n(1)) + std::abs(n(2));
    if (l1 == 0) return packSnorm2x16(0, 0);
    float x = n(0) / l1, y = n(1) / l1;
    if (n(2) < 0)
    {
        const float ox = (1 - std::abs(y)) * (x >= 0 ? 1.0f : -1.0f);
        const float oy = (1 - std::abs(x)) * (y >= 0 ? 1.0f : -1.0f);
        x              = ox;
        y              = oy;
    }
    return packSnorm2x16(x, y);
}

vec3 unpackNormalOct(uint32_t v)
{
    // decodeOct of shader/vulkan/scene/instance.vert
    const vec2 e = unpackSnorm2x16(v);
    vec3 n(e(0), e(1), 1 - std::abs(e(0)) - std::abs(e(1)));
    if (n(2) < 0)
    {
        const float x = (1 - std::abs(n(1))) * (n(0) >= 0 ? 1.0f : -1.0f);
        const float y = (1 - std::abs(n(0))) * (n(1) >= 0 ? 1.0f : -1.0f);
        n(0)          = x;
        n(1)          = y;
    }
    return n.normalized();
}

void StaticInstance::setRotation(const quat& q)
{
    const quat n = q.normalized();
    rot01        = packSnorm2x16(n.x(), n.y());
    rot23        = packSnorm2x16(n.z(), n.w());
}

quat StaticInstance::getRotation() const
{
    const vec2 a = unpackSnorm2x16(rot01);
    const vec2 b = unpackSnorm2x16(rot23);
    quat q(b(1), a(0), a(1), b(0));  // (w, x, y, z)
    return q.normalized();
}

}  // namespace Vulkan
}  // namespace Saiga
