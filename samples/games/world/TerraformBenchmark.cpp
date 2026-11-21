/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "games/world/TerraformBenchmark.h"

#include "saiga/engine/net/Replication.h"
#include "saiga/engine/terrain/TerrainEditStore.h"
#include "saiga/engine/terrain/TerrainMessages.h"
#include "saiga/engine/terrain/TerrainStore.h"
#include "saiga/engine/terrain/TerrainSystem.h"
#include "saiga/vulkan/world/WorldBenchmark.h"

#include <chrono>
#include <cmath>
#include <sstream>

namespace Saiga
{
namespace Game
{
using namespace Engine;

namespace
{
constexpr double kPi = 3.14159265358979323846;
// Brush b = 2c + j: the op and the radius (the table of the S22 plan)
struct BrushSpec
{
    TerrainBrush::Op op;
    double radius;
};
constexpr BrushSpec kBrushes[8] = {
    {TerrainBrush::Add, 4},  {TerrainBrush::Flatten, 12}, {TerrainBrush::Subtract, 8},  {TerrainBrush::Smooth, 16},
    {TerrainBrush::Add, 16}, {TerrainBrush::Flatten, 8},  {TerrainBrush::Subtract, 12}, {TerrainBrush::Smooth, 4},
};
Vec3 corner(int c)
{
    return Vec3(3840 + 256 * (c % 2), 0, 3840 + 256 * (c / 2));
}
json11::Json stat(const std::vector<double>& v)
{
    const auto s = Vulkan::WorldBenchmark::summarize(v);
    return json11::Json::object{{"median", s.median}, {"p95", s.p95}};
}
}  // namespace

TerraformBenchmark::TerraformBenchmark(const World& world)
{
    for (int c = 0; c < 4; ++c) targets[size_t(c)] = world.terrainHeight(0, corner(c).x(), corner(c).z());
}

void TerraformBenchmark::fail(const std::string& what)
{
    if (errors.size() < 20) errors.push_back(what);
}

void TerraformBenchmark::begin(const World& world)
{
    hookStart  = world.terrain().baseHookSamples();
    masksStart = world.terrain().stats().maskRecomputes;
    for (int z = 14; z <= 16; ++z)
        for (int x = 14; x <= 16; ++x)
        {
            RegionId r;
            r.space = 0;
            r.level = 0;
            r.x     = x;
            r.z     = z;
            if (!world.terrain().store(0).tile(r))
                fail("the level-0 tile of chunk (" + std::to_string(x) + ", " + std::to_string(z) +
                     ") is not resident at the first measured frame");
        }
}

void TerraformBenchmark::edit(World& world)
{
    // The step that just ran decides the brush
    const int b = int((world.stepIndex() - 1) % 8);
    const int c = b / 2, j = b % 2;
    const BrushSpec spec = kBrushes[b];
    const int n          = editsPerBrush[size_t(b)]++;
    const double orbit   = 0.5 * spec.radius;
    const double angle   = j * kPi + double(n) / orbit;  // 1 m of arc per edit
    TerrainBrush brush;
    brush.op       = spec.op;
    brush.radius   = spec.radius;
    brush.strength = (spec.op == TerrainBrush::Add || spec.op == TerrainBrush::Subtract) ? 0.25 : 0.5;
    brush.center   = corner(c) + Vec3(orbit * std::cos(angle), 0, orbit * std::sin(angle));
    if (spec.op == TerrainBrush::Flatten) brush.center(1) = targets[size_t(c)];

    const TerrainEditStore::Stats before = world.terrainEdits().stats();
    const size_t listBefore              = world.replication().hostStateList().size();
    const auto t0                        = std::chrono::steady_clock::now();
    world.editTerrain(0, brush);
    const auto t1 = std::chrono::steady_clock::now();
    editMs.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
    const TerrainEditStore::Stats& after = world.terrainEdits().stats();
    const uint64_t chunkCount            = after.chunksEdited - before.chunksEdited;
    const int op                         = int(spec.op);
    noops[size_t(op)] += after.noopEdits - before.noopEdits;
    chunksPerOp[size_t(op)] += chunkCount;
    editsPerOp[size_t(op)]++;
    chunks += chunkCount;
    samples += after.changedSamples - before.changedSamples;
    edits++;
    if ((spec.op == TerrainBrush::Add || spec.op == TerrainBrush::Subtract) && chunkCount != 4)
        fail("edit " + std::to_string(edits) + " (brush " + std::to_string(b) + ") changed " +
             std::to_string(chunkCount) + " chunks instead of 4");
    // The encoding of the new host state list entry, as phase 8 would send it to one client
    const auto& list = world.replication().hostStateList();
    if (list.size() > listBefore)
    {
        const auto e0 = std::chrono::steady_clock::now();
        scratch.clear();
        encodeTerrainModification(world.replication().epoch(), list.back().space, list.back().parts, scratch);
        encodeMs.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - e0).count());
        encodedBytes += scratch.size();
    }
    else
        encodeMs.push_back(0);
}

void TerraformBenchmark::frameDone(const World& world, double uploadBytes)
{
    const TerrainSystem::Stats& s = world.terrain().stats();
    visualMs.push_back(s.lastVisualMs);
    patched.push_back(double(s.patchedSamplesLastUpdate));
    uploads.push_back(uploadBytes);
    if (s.pending != 0) fail("terrain requests pending in a measured frame (" + std::to_string(s.pending) + ")");
    if (s.uploadsLastUpdate != 0) fail("a tile was sent in a measured frame (setTerrainTile)");
}

void TerraformBenchmark::end(const World& world)
{
    hookEnd    = world.terrain().baseHookSamples();
    masksEnd   = world.terrain().stats().maskRecomputes;
    storeBytes = world.terrainEdits().memoryBytes();
    if (hookEnd != hookStart) fail(std::to_string(hookEnd - hookStart) + " base samples read from the height hook");
    if (chunks < 1200) fail("chunks_edited " + std::to_string(chunks) + " < 1200");
}

void TerraformBenchmark::addResult(json11::Json::object& r) const
{
    // 600 measured frames at one step per frame: 10 s of simulation at 60 steps/s
    const double seconds = double(Vulkan::WorldBenchmark::measuredFrames) / 60.0;
    r["edits"]           = edits;
    r["noop_edits"]      = json11::Json::object{{"add", double(noops[0])},
                                                {"subtract", double(noops[1])},
                                                {"flatten", double(noops[2])},
                                                {"smooth", double(noops[3])}};
    r["chunks_edited"]   = double(chunks);
    r["chunks_per_edit"] =
        json11::Json::object{{"flatten", editsPerOp[2] ? double(chunksPerOp[2]) / double(editsPerOp[2]) : 0.0},
                             {"smooth", editsPerOp[3] ? double(chunksPerOp[3]) / double(editsPerOp[3]) : 0.0}};
    r["samples_changed"]       = double(samples);
    r["terrain_edit_ms"]       = stat(editMs);
    r["base_hook_samples"]     = double(hookEnd - hookStart);
    r["terrain_encode_ms"]     = stat(encodeMs);
    r["edit_bytes_per_second"] = double(encodedBytes) / seconds;
    r["terrain_visual_ms"]     = stat(visualMs);
    r["patched_samples"]       = stat(patched);
    r["upload_bytes"]          = stat(uploads);
    r["mask_recomputes"]       = double(masksEnd - masksStart);
    r["edit_store_bytes"]      = double(storeBytes);
    json11::Json::array f;
    for (const std::string& e : errors) f.push_back(e);
    r["terraform_errors"] = f;
}

std::string TerraformBenchmark::summary() const
{
    auto ms = [](const std::vector<double>& v)
    {
        const auto s = Vulkan::WorldBenchmark::summarize(v);
        std::ostringstream o;
        o << s.median << " / " << s.p95;
        return o.str();
    };
    std::ostringstream o;
    o << "  terraform: " << edits << " edits, " << chunks << " chunks, " << samples << " samples changed; noop edits "
      << noops[0] << " / " << noops[1] << " / " << noops[2] << " / " << noops[3]
      << " (add / subtract / flatten / smooth)"
      << "\n";
    o << "  terrain_edit_ms median / p95 " << ms(editMs) << " (budget p95 <= 1 ms), terrain_encode_ms " << ms(encodeMs)
      << ", edit bytes/s " << double(encodedBytes) / 10.0 << " (report: 256 kbit/s = 32000 B/s per client)\n";
    o << "  terrain_visual_ms median / p95 " << ms(visualMs) << " (budget p95 <= 1 ms), patched samples " << ms(patched)
      << ", upload bytes " << ms(uploads) << ", hook samples " << (hookEnd - hookStart) << ", mask recomputes "
      << (masksEnd - masksStart) << ", edit store " << storeBytes << " bytes\n";
    for (const std::string& e : errors) o << "  Benchmark error: " << e << "\n";
    return o.str();
}

}  // namespace Game
}  // namespace Saiga
