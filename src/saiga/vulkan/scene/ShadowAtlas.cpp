/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vulkan/scene/ShadowAtlas.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace Saiga
{
namespace Vulkan
{
using Stage  = vk::PipelineStageFlagBits2;
using Access = vk::AccessFlagBits2;

namespace
{
constexpr vk::Format kAtlasFormat = vk::Format::eD16Unorm;
// The hysteresis of the tile selection (B4): a light that owns tiles counts this much more
constexpr double kHysteresis = 1.2;

vk::Buffer handle(Buffer& b)
{
    Memory::SafeAccessor acc(*b.memoryLocation());
    return b.memoryLocation()->data;
}

uint64_t mix(uint64_t h, uint64_t v)
{
    h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
    return h;
}
uint64_t bits(double d)
{
    uint64_t u;
    std::memcpy(&u, &d, sizeof(u));
    return u;
}
uint64_t bits(float f)
{
    uint32_t u;
    std::memcpy(&u, &f, sizeof(u));
    return u;
}
}  // namespace

void ShadowAtlas::init(VulkanBase& b)
{
    destroy();
    base = &b;
    for (Buffer& t : tables)
        t.createBuffer(*base, sizeof(GpuTile) * 64, vk::BufferUsageFlagBits::eStorageBuffer,
                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
}

void ShadowAtlas::destroy()
{
    if (!base) return;
    release();
    for (Buffer& t : tables) t.destroy();
    base = nullptr;
}

void ShadowAtlas::release()
{
    if (!base) return;
    auto device = base->device;
    for (vk::ImageView& v : layerViews)
    {
        auto view = v;
        if (view) base->deferDelete([device, view]() { device.destroyImageView(view); });
        v = nullptr;
    }
    // Texture::destroy releases the image after the frames in flight
    if (image) image->destroy();
    image.reset();
    res = tileTexels = 0;
    tileState.clear();
    tileKeys.clear();
    owners.clear();
}

void ShadowAtlas::ensure(int resolution, int tile)
{
    if (image && res == resolution && tileTexels == tile) return;
    release();
    image = std::make_unique<Texture2DArray>();
    image->create(*base, uint32_t(resolution), uint32_t(resolution), 2, kAtlasFormat,
                  vk::ImageUsageFlagBits::eDepthStencilAttachment | vk::ImageUsageFlagBits::eSampled);
    res        = resolution;
    tileTexels = tile;
    for (uint32_t l = 0; l < 2; ++l)
    {
        vk::ImageViewCreateInfo info({}, image->image(), vk::ImageViewType::e2D, kAtlasFormat, {},
                                     vk::ImageSubresourceRange(vk::ImageAspectFlagBits::eDepth, 0, 1, l, 1));
        layerViews[l] = base->device.createImageView(info);
    }
    // Both layers lit (1) and readable
    auto& q               = base->mainQueue;
    vk::CommandBuffer cmd = q.commandPool.createAndBeginOneTimeBuffer();
    const vk::ImageSubresourceRange range(vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 2);
    vk::ImageMemoryBarrier2 toDst(Stage::eNone, Access::eNone, Stage::eClear, Access::eTransferWrite,
                                  vk::ImageLayout::eUndefined, vk::ImageLayout::eTransferDstOptimal,
                                  VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, image->image(), range);
    cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, toDst));
    cmd.clearDepthStencilImage(image->image(), vk::ImageLayout::eTransferDstOptimal,
                               vk::ClearDepthStencilValue(1.0f, 0), range);
    vk::ImageMemoryBarrier2 toRead(Stage::eClear, Access::eTransferWrite, Stage::eAllCommands,
                                   Access::eShaderSampledRead, vk::ImageLayout::eTransferDstOptimal,
                                   vk::ImageLayout::eShaderReadOnlyOptimal, VK_QUEUE_FAMILY_IGNORED,
                                   VK_QUEUE_FAMILY_IGNORED, image->image(), range);
    cmd.pipelineBarrier2(vk::DependencyInfo({}, nullptr, nullptr, toRead));
    cmd.end();
    q.submitAndWait(cmd);
    q.commandPool.freeCommandBuffer(cmd);
    image->setImageLayout(vk::ImageLayout::eShaderReadOnlyOptimal);
    const int n = tilesPerSide() * tilesPerSide();
    tileState.assign(size_t(n), AtlasTileInfo());
    tileKeys.assign(size_t(n), 0);
}

void ShadowAtlas::markDirty(const Vec3& lo, const Vec3& hi)
{
    // A long list (a terrain stream) collapses into one invalidation
    if (dirty.size() > 4096)
        allDirty = true;
    else
        dirty.emplace_back(lo, hi);
}

void ShadowAtlas::retain(const std::function<bool(uint32_t)>& keep)
{
    std::vector<uint32_t> drop;
    for (const auto& o : owners)
        if (!keep(o.first)) drop.push_back(o.first);
    for (uint32_t h : drop)
    {
        for (int t : owners[h].tiles)
        {
            tileState[size_t(t)].owner       = 0;
            tileState[size_t(t)].staticValid = false;
            tileState[size_t(t)].staticFrame = 0;
            tileKeys[size_t(t)]              = 0;
        }
        owners.erase(h);
    }
}

void ShadowAtlas::faceView(const LightData& d, const Vec3& lightRel, int face, mat4& viewProj,
                           std::array<Vec4, 6>& planes, double& tanHalf)
{
    static const Vec3 axes[6] = {Vec3(1, 0, 0),  Vec3(-1, 0, 0), Vec3(0, 1, 0),
                                 Vec3(0, -1, 0), Vec3(0, 0, 1),  Vec3(0, 0, -1)};
    Vec3 F;
    if (d.kind == LightKind::Spot)
    {
        F                = d.direction.cast<double>();
        F                = F.norm() > 1e-9 ? Vec3(F.normalized()) : Vec3(0, -1, 0);
        const double co  = std::clamp(double(d.cosOuter), 0.05, 1.0);
        const double tan = std::sqrt(1 - co * co) / co;
        // A small margin over the cone (the receivers' filter at its edge)
        tanHalf = std::min(tan * 1.05 + 0.01, 11.43);
    }
    else
    {
        F       = axes[std::clamp(face, 0, 5)];
        tanHalf = 1.0;
    }
    const Vec3 up      = std::abs(F(1)) > 0.9 ? Vec3(0, 0, -1) : Vec3(0, 1, 0);
    const Vec3 R       = F.cross(up).normalized();
    const Vec3 U       = R.cross(F);
    const double range = std::max(double(d.range), 1e-3);
    const double nearP = std::max(0.02, range * 1e-3);
    // The view (rows R, U, -F: right-handed like the main camera) and a standard perspective (depth 0 near, 1 far;
    // the atlas stores the linear depth of the fragment shader, z only clips)
    Mat4 V             = Mat4::Identity();
    const Vec3 rows[3] = {R, U, -F};
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j) V(i, j) = rows[i](j);
        V(i, 3) = -rows[i].dot(lightRel);
    }
    Mat4 P   = Mat4::Zero();
    P(0, 0)  = 1.0 / tanHalf;
    P(1, 1)  = -1.0 / tanHalf;  // Vulkan's y axis points down (as the main camera)
    P(2, 2)  = range / (nearP - range);
    P(2, 3)  = nearP * range / (nearP - range);
    P(3, 2)  = -1;
    viewProj = (P * V).cast<float>();
    // The culling planes (camera-relative, inward)
    auto plane = [&](const Vec3& n, double w) { return Vec4(n(0), n(1), n(2), w); };
    auto side  = [&](const Vec3& n)
    {
        const Vec3 m = n.normalized();
        return plane(m, -m.dot(lightRel));
    };
    planes = {side(R + F * tanHalf),
              side(-R + F * tanHalf),
              side(U + F * tanHalf),
              side(-U + F * tanHalf),
              plane(F, -F.dot(lightRel) - nearP),
              plane(-F, F.dot(lightRel) + range)};
}

uint64_t ShadowAtlas::faceKey(const LightInput& l, int face)
{
    uint64_t h = 1469598103934665603ull;
    h          = mix(h, l.handle);
    h          = mix(h, uint64_t(face));
    for (int i = 0; i < 3; ++i) h = mix(h, bits(l.world(i)));
    h = mix(h, uint64_t(l.data.kind));
    h = mix(h, bits(l.data.range));
    if (l.data.kind == LightKind::Spot)
    {
        for (int i = 0; i < 3; ++i) h = mix(h, bits(l.data.direction(i)));
        h = mix(h, bits(l.data.cosOuter));
    }
    return mix(h, l.terrainKey);
}

void ShadowAtlas::update(const std::vector<LightInput>& lights, const Vec3& camera, int facesPerFrame, uint64_t frame,
                         std::vector<Face>& faces)
{
    (void)camera;
    faces.clear();
    stats               = AtlasStatistics();
    stats.facesPerFrame = facesPerFrame;
    if (!image)
    {
        dirty.clear();
        allDirty = false;
        return;
    }
    stats.on         = true;
    stats.resolution = res;
    stats.tile       = tileTexels;
    const int total  = tilesPerSide() * tilesPerSide();
    stats.tilesTotal = uint32_t(total);

    // 1. The selection by tiles (importance, hysteresis for the owners; ties by handle)
    const size_t n = lights.size();
    std::vector<size_t> order(n);
    for (size_t i = 0; i < n; ++i) order[i] = i;
    auto score = [&](size_t i) { return lights[i].importance * (owners.count(lights[i].handle) ? kHysteresis : 1.0); };
    std::sort(order.begin(), order.end(),
              [&](size_t a, size_t b)
              {
                  const double sa = score(a), sb = score(b);
                  if (sa != sb) return sa > sb;
                  return lights[a].handle < lights[b].handle;
              });
    std::vector<bool> selected(n, false);
    int budget = total;
    for (size_t i : order)
    {
        const int need = faceCount(lights[i].data);
        if (need <= budget)
        {
            selected[i] = true;
            budget -= need;
        }
    }
    std::unordered_map<uint32_t, size_t> inputOf;
    for (size_t i = 0; i < n; ++i) inputOf[lights[i].handle] = i;
    auto freeOwner = [&](uint32_t handle)
    {
        auto it = owners.find(handle);
        if (it == owners.end()) return;
        for (int t : it->second.tiles)
        {
            // Stale dynamic content of the old light is cleared below when the tile is reused
            tileState[size_t(t)].owner       = 0;
            tileState[size_t(t)].staticValid = false;
            tileState[size_t(t)].staticFrame = 0;
            tileKeys[size_t(t)]              = 0;
        }
        owners.erase(it);
    };
    // 2. Candidates that lost their tiles free them (non-candidates stay cached until their tiles are needed)
    std::vector<uint32_t> losers;
    for (const auto& o : owners)
    {
        auto in = inputOf.find(o.first);
        if (in != inputOf.end() && !selected[in->second]) losers.push_back(o.first);
    }
    for (uint32_t h : losers) freeOwner(h);
    // 3. The selected lights without tiles take free tiles (evicting the oldest cached non-candidates)
    for (size_t i : order)
    {
        if (!selected[i] || owners.count(lights[i].handle)) continue;
        const int need = faceCount(lights[i].data);
        std::vector<int> freeTiles;
        for (int t = 0; t < total && int(freeTiles.size()) < need; ++t)
            if (tileState[size_t(t)].owner == 0) freeTiles.push_back(t);
        while (int(freeTiles.size()) < need)
        {
            uint32_t oldest = 0;
            uint64_t age    = ~uint64_t(0);
            for (const auto& o : owners)
                if (!inputOf.count(o.first) && o.second.lastUsed < age)
                {
                    age    = o.second.lastUsed;
                    oldest = o.first;
                }
            if (!oldest) break;
            freeOwner(oldest);
            freeTiles.clear();
            for (int t = 0; t < total && int(freeTiles.size()) < need; ++t)
                if (tileState[size_t(t)].owner == 0) freeTiles.push_back(t);
        }
        if (int(freeTiles.size()) < need) continue;
        Owner& o = owners[lights[i].handle];
        o.tiles  = freeTiles;
        for (int k = 0; k < need; ++k)
        {
            AtlasTileInfo& t = tileState[size_t(freeTiles[size_t(k)])];
            t.owner          = lights[i].handle;
            t.face           = k;
            t.staticValid    = false;
            t.staticFrame    = 0;
        }
    }
    // 4. Invalidation: the keys, the dirty bounds; the queue
    struct Job
    {
        size_t light;
        int face, tile;
        uint64_t age;
    };
    std::vector<Job> dynamicJobs, staticJobs;
    for (size_t i : order)
    {
        auto it = owners.find(lights[i].handle);
        if (it == owners.end() || !selected[i])
        {
            if (faceCount(lights[i].data) > 0) stats.lightsWithoutTiles++;
            continue;
        }
        stats.lightsWithTiles++;
        Owner& o            = it->second;
        o.lastUsed          = frame;
        const LightInput& l = lights[i];
        const double r      = double(l.data.range);
        bool dirtyHit       = allDirty;
        for (size_t k = 0; k < dirty.size() && !dirtyHit; ++k)
        {
            const Vec3 d = (dirty[k].first - l.world).cwiseMax(Vec3::Zero()).cwiseMax(l.world - dirty[k].second);
            dirtyHit     = d.squaredNorm() <= r * r;
        }
        for (int k = 0; k < int(o.tiles.size()); ++k)
        {
            const int t        = o.tiles[size_t(k)];
            AtlasTileInfo& ts  = tileState[size_t(t)];
            const uint64_t key = faceKey(l, k);
            if (key != tileKeys[size_t(t)] || dirtyHit) ts.staticValid = false;
            tileKeys[size_t(t)] = key;
            if (!ts.staticValid)
                staticJobs.push_back({i, k, t, ts.staticFrame});
            else
                stats.hits++;
            if (l.dynamics) dynamicJobs.push_back({i, k, t, ts.dynamicFrame});
        }
    }
    dirty.clear();
    allDirty = false;
    // Dynamic tiles whose light has no dynamic instance in range any more (or a new owner): cleared once
    for (int t = 0; t < total; ++t)
    {
        AtlasTileInfo& ts = tileState[size_t(t)];
        if (!ts.dynamicContent) continue;
        auto in         = inputOf.find(ts.owner);
        const bool keep = ts.owner != 0 && in != inputOf.end() && lights[in->second].dynamics;
        if (keep) continue;
        Face f;
        f.tile      = t;
        f.layer     = 1;
        f.clearOnly = true;
        faces.push_back(f);
    }
    // The queue: dynamic faces first (importance), then the invalid static faces by importance and age
    std::stable_sort(staticJobs.begin(), staticJobs.end(),
                     [&](const Job& a, const Job& b)
                     {
                         if (a.light != b.light)
                         {
                             const double ia = lights[a.light].importance, ib = lights[b.light].importance;
                             if (ia != ib) return ia > ib;
                         }
                         return a.age < b.age;
                     });
    int budgetFaces = std::max(facesPerFrame, 0);
    auto emit       = [&](const Job& j, int layer)
    {
        const LightInput& l = lights[j.light];
        Face f;
        f.tile  = j.tile;
        f.layer = layer;
        f.face  = j.face;
        f.light = j.light;
        double tanHalf;
        const Vec3 rel = l.world - camera;
        faceView(l.data, rel, j.face, f.viewProj, f.planes, tanHalf);
        f.lightPush = vec4(float(rel(0)), float(rel(1)), float(rel(2)), 1.0f / std::max(l.data.range, 1e-3f));
        faces.push_back(f);
    };
    for (const Job& j : dynamicJobs)
    {
        if (budgetFaces == 0) break;
        emit(j, 1);
        budgetFaces--;
        stats.dynamicRenders++;
    }
    for (const Job& j : staticJobs)
    {
        if (budgetFaces == 0)
        {
            stats.queue++;
            continue;
        }
        emit(j, 0);
        budgetFaces--;
        stats.staticRenders++;
    }
    for (const Face& f : faces) stats.dynamicClears += f.clearOnly ? 1 : 0;
    uint32_t used = 0;
    for (const AtlasTileInfo& t : tileState) used += t.owner ? 1 : 0;
    stats.tilesUsed   = used;
    stats.memoryBytes = memoryBytes();
}

void ShadowAtlas::rendered(const std::vector<Face>& faces, uint64_t frame)
{
    for (const Face& f : faces)
    {
        AtlasTileInfo& t = tileState[size_t(f.tile)];
        if (f.clearOnly)
        {
            t.dynamicContent = false;
            continue;
        }
        if (f.layer == 0)
        {
            t.staticValid = true;
            t.staticFrame = frame;
        }
        else
        {
            t.dynamicContent = true;
            t.dynamicFrame   = frame;
        }
    }
}

void ShadowAtlas::writeTable(uint32_t frameIndex, const std::vector<LightInput>& lights, const Vec3& camera,
                             std::vector<int>& firstEntry, std::vector<bool>& ready)
{
    firstEntry.assign(lights.size(), -1);
    ready.assign(lights.size(), false);
    std::vector<GpuTile> entries;
    const int perSide = std::max(tilesPerSide(), 1);
    for (size_t i = 0; i < lights.size() && image; ++i)
    {
        auto it = owners.find(lights[i].handle);
        if (it == owners.end() || it->second.lastUsed == 0) continue;
        const Owner& o = it->second;
        // Only the lights selected this frame (update set lastUsed)
        bool complete = true;
        firstEntry[i] = int(entries.size());
        for (int k = 0; k < int(o.tiles.size()); ++k)
        {
            const int t = o.tiles[size_t(k)];
            GpuTile g;
            std::array<Vec4, 6> planes;
            double tanHalf;
            faceView(lights[i].data, lights[i].world - camera, k, g.viewProj, planes, tanHalf);
            const float s = float(tileTexels) / float(res);
            g.rect        = vec4(float(t % perSide) * s, float(t / perSide) * s, s, s);
            g.params      = vec4(float(2 * tanHalf / tileTexels), 1.0f / float(tileTexels),
                            tileState[size_t(t)].dynamicContent ? 1.0f : 0.0f, 0);
            complete      = complete && tileState[size_t(t)].staticFrame > 0;
            entries.push_back(g);
        }
        ready[i] = complete;
    }
    Buffer& b          = tables[frameIndex];
    const size_t bytes = std::max<size_t>(entries.size(), 1) * sizeof(GpuTile);
    if (b.size() < bytes)
        b.createBuffer(*base, std::max(bytes, size_t(2 * b.size())), vk::BufferUsageFlagBits::eStorageBuffer,
                       vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent);
    if (!entries.empty()) std::memcpy(b.getMappedPointer(), entries.data(), entries.size() * sizeof(GpuTile));
}

vk::DescriptorBufferInfo ShadowAtlas::tableInfo(uint32_t frameIndex)
{
    return vk::DescriptorBufferInfo(handle(tables[frameIndex]), 0, VK_WHOLE_SIZE);
}

std::vector<int> ShadowAtlas::tilesOf(uint32_t handle) const
{
    auto it = owners.find(handle);
    return it == owners.end() ? std::vector<int>() : it->second.tiles;
}

uint64_t ShadowAtlas::memoryBytes() const
{
    return uint64_t(res) * uint64_t(res) * 2 * 2 + tables[0].size() + tables[1].size();
}

}  // namespace Vulkan
}  // namespace Saiga
