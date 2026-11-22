/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/scene/LocalLights.h"

#include "saiga/core/util/assert.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace Saiga
{
namespace Vulkan
{
namespace
{
constexpr double kPi = 3.14159265358979323846;
}

void LocalLights::init(VulkanBase& b)
{
    destroy();
    base = &b;
    for (Buffer& buf : buffers)
    {
        buf.createBuffer(*base, kHeaderBytes + 64 * sizeof(GpuLight), vk::BufferUsageFlagBits::eStorageBuffer,
                         vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
        std::memset(buf.getMappedPointer(), 0, kHeaderBytes);
    }
}

void LocalLights::destroy()
{
    if (!base) return;
    for (Buffer& buf : buffers) buf.destroy();
    base = nullptr;
}

float LocalLights::candela(const LightData& d)
{
    return float(double(d.lumen) / (d.kind == LightKind::Spot ? kPi : 4 * kPi));
}

float LocalLights::flickerFactor(const LightFlicker& f, float phase)
{
    if (!(f.amplitude > 0) || !(f.frequency > 0)) return 1.0f;
    // Integer multiples of the base phase (10 rad/s, EnvironmentState::phases[6], wrapped at 2 pi): continuous through
    // the wrap; the seed shifts the phases
    const double k = std::max(1.0, std::round(double(f.frequency) * 2 * kPi / 10.0));
    const double s = double(f.seed % 1024u) * 0.00613592315;  // seed * 2 pi / 1024
    const double p = double(phase);
    const double n = 0.55 * std::sin(k * p + s) + 0.3 * std::sin((2 * k + 1) * p + 2.3 * s) +
                     0.15 * std::sin((3 * k + 2) * p + 4.1 * s);
    const double fl = 1.0 + double(f.amplitude) * n;
    return float(std::max(0.0, fl));
}

double LocalLights::importance(const LightData& d, double distance)
{
    const double r = std::max(distance, double(d.range));
    return double(d.priority) * double(d.lumen) / (r * r);
}

int LocalLights::frameIndexOf(LightHandle h) const
{
    auto it = frameIndexByHandle.find(h.id);
    return it == frameIndexByHandle.end() ? -1 : it->second;
}

void LocalLights::prepareFrame(uint32_t frameIndex, const FrameInput& in)
{
    frameInfo.clear();
    records.clear();
    frameIndexByHandle.clear();
    highlighted     = -1;
    stats           = LocalLightStatistics();
    stats.on        = in.on;
    stats.submitted = uint32_t(table.size());
    if (in.on && in.valid)
    {
        struct Candidate
        {
            size_t index;
            double distance, importance;
        };
        std::vector<Candidate> cand;
        for (size_t i = 0; i < table.size(); ++i)
        {
            const Light& l = table[i];
            if (l.chunk.space != in.space || !(l.data.range > 0) || !(l.data.lumen > 0)) continue;
            const Vec3 rel        = l.worldPosition() - in.camera;
            const double distance = rel.norm();
            const double r        = double(l.data.range);
            if (distance - r > in.clusterFar) continue;
            bool inside = true;
            for (const Vec4& p : in.planes)
                inside = inside && p(0) * rel(0) + p(1) * rel(1) + p(2) * rel(2) + p(3) >= -r;
            if (!inside) continue;
            cand.push_back({i, distance, importance(l.data, distance)});
        }
        stats.candidates = uint32_t(cand.size());
        std::sort(cand.begin(), cand.end(),
                  [&](const Candidate& a, const Candidate& b)
                  {
                      if (a.importance != b.importance) return a.importance > b.importance;
                      return table[a.index].handle.id < table[b.index].handle.id;
                  });
        const size_t cap = size_t(std::clamp(in.maxLights, 1, 4096));
        if (cand.size() > cap)
        {
            stats.overBudget = uint32_t(cand.size() - cap);
            cand.resize(cap);
        }
        for (const Candidate& c : cand)
        {
            const Light& l     = table[c.index];
            const LightData& d = l.data;
            GpuLight g;
            const Vec3 rel = l.worldPosition() - in.camera;
            g.posRange     = vec4(float(rel(0)), float(rel(1)), float(rel(2)), d.range);
            const vec3 dir = d.direction.norm() > 1e-6f ? vec3(d.direction.normalized()) : vec3(0, -1, 0);
            g.dirKind      = vec4(dir(0), dir(1), dir(2), d.kind == LightKind::Spot ? 1.0f : 0.0f);
            const float cd = candela(d) * flickerFactor(d.flicker, in.flickerPhase);
            g.intensity    = vec4(d.color(0) * cd, d.color(1) * cd, d.color(2) * cd, -1.0f);
            const float co = std::min(d.cosOuter, d.cosInner);
            const float ci = std::max(d.cosInner, co + 1e-4f);
            g.cone         = vec4(co, 1.0f / (ci - co), 0, 0);
            FrameLightInfo info;
            info.handle     = l.handle.id;
            info.index      = uint32_t(records.size());
            info.distance   = c.distance;
            info.importance = c.importance;
            if (l.handle.id == in.highlight) highlighted = int(records.size());
            frameIndexByHandle[l.handle.id] = int(records.size());
            records.push_back(g);
            frameInfo.push_back(info);
            if (d.castShadows) stats.shadowed++;
        }
        stats.visible = uint32_t(records.size());
    }
    writeRecords(frameIndex);
}

void LocalLights::writeRecords(uint32_t frameIndex)
{
    Buffer& buf        = buffers[frameIndex];
    const size_t bytes = kHeaderBytes + std::max<size_t>(records.size(), 1) * sizeof(GpuLight);
    if (buf.size() < bytes)
    {
        buf.createBuffer(*base, std::max(bytes, 2 * buf.size()), vk::BufferUsageFlagBits::eStorageBuffer,
                         vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    }
    auto* dst                = static_cast<uint8_t*>(buf.getMappedPointer());
    const uint32_t header[4] = {uint32_t(records.size()), 0, 0, 0};
    std::memcpy(dst, header, sizeof(header));
    if (!records.empty()) std::memcpy(dst + kHeaderBytes, records.data(), records.size() * sizeof(GpuLight));
    stats.memoryBytes = buffers[0].size() + buffers[1].size();
}

vk::DescriptorBufferInfo LocalLights::bufferInfo(uint32_t frameIndex) const
{
    Buffer& buf = const_cast<Buffer&>(buffers[frameIndex]);
    Memory::SafeAccessor acc(*buf.memoryLocation());
    return vk::DescriptorBufferInfo(buf.memoryLocation()->data, 0, VK_WHOLE_SIZE);
}

LightHandle LocalLights::add(const ChunkId& chunk, const LightData& light)
{
    const LightHandle h{nextHandle++};
    indexOf[h.id] = table.size();
    table.push_back(Light{h, chunk, light, 0});
    changes++;
    return h;
}

void LocalLights::set(LightHandle h, const ChunkId& chunk, const LightData& light)
{
    auto it = indexOf.find(h.id);
    SAIGA_ASSERT(it != indexOf.end(), "RenderScene::setLight: unknown light handle");
    Light& l = table[it->second];
    SAIGA_ASSERT(l.list == 0, "RenderScene::setLight: the light belongs to a list (addLights)");
    l.chunk = chunk;
    l.data  = light;
    changes++;
}

void LocalLights::erase(LightHandle h)
{
    auto it = indexOf.find(h.id);
    SAIGA_ASSERT(it != indexOf.end(), "RenderScene::removeLight: unknown light handle");
    const size_t i = it->second;
    indexOf.erase(it);
    if (i + 1 != table.size())
    {
        table[i]                    = table.back();
        indexOf[table[i].handle.id] = i;
    }
    table.pop_back();
    changes++;
}

void LocalLights::remove(LightHandle h)
{
    auto it = indexOf.find(h.id);
    SAIGA_ASSERT(it != indexOf.end() && table[it->second].list == 0,
                 "RenderScene::removeLight: unknown light handle (or one of a list)");
    erase(h);
}

LightListHandle LocalLights::addList(const ChunkId& chunk, const std::vector<LightData>& lights)
{
    const LightListHandle list{nextList++};
    std::vector<LightHandle>& handles = lists[list.id];
    for (const LightData& d : lights)
    {
        const LightHandle h = add(chunk, d);
        table.back().list   = list.id;
        handles.push_back(h);
    }
    return list;
}

void LocalLights::removeList(LightListHandle list)
{
    auto it = lists.find(list.id);
    SAIGA_ASSERT(it != lists.end(), "RenderScene::removeLights: unknown light list");
    for (LightHandle h : it->second) erase(h);
    lists.erase(it);
}

const LocalLights::Light* LocalLights::find(LightHandle h) const
{
    auto it = indexOf.find(h.id);
    return it == indexOf.end() ? nullptr : &table[it->second];
}

}  // namespace Vulkan
}  // namespace Saiga
