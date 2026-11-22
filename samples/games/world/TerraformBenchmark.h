/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// TerraformBenchmark (docs/engine_design.md §17.7; S22): the edit schedule of game_world --benchmark terraform. 8
// brushes, two on each of the 4 chunk corners C_c = (3840 + 256 (c % 2), 3840 + 256 (c / 2)): a roughening one (Add /
// Subtract, 0.25 m) and a levelling one (Flatten to the start height of C_c / Smooth, 0.5); brush b = 2c + j orbits
// C_c on a circle of 0.5 r_b, starting at j * 180 degrees, 1 m of arc per edit. One edit per measured frame after the
// frame's step (brush b after steps with step % 8 == b), timed per World::editTerrain call; the host state list's new
// entry is encoded with the phase-8 encoder (timed separately, nothing sent). Collects the per-frame terrain numbers
// and checks the assertions (4 chunks per Add/Subtract edit, chunks_edited >= 1200, no hook reads, no tile sends or
// pending requests in the measured frames, the level-0 tiles of the brush areas resident).

#pragma once

#include "saiga/core/util/json11.hpp"
#include "saiga/engine/world/World.h"

#include <array>
#include <string>
#include <vector>

namespace Saiga
{
namespace Game
{
class TerraformBenchmark
{
   public:
    // The floor of game_world's residencyRange0 rule in this benchmark (game policy; within 48 tiles per level)
    static constexpr double kResidencyRange0 = 600;

    // The flatten targets: terrainHeight of the corners at the start
    explicit TerraformBenchmark(const Engine::World& world);
    // The first measured frame (before its edit): the level-0 tiles of chunks (14..16)^2 are resident
    void begin(const Engine::World& world);
    // The edit of this measured frame (after its step, before its updateView)
    void edit(Engine::World& world);
    // The terrain numbers of a measured frame's update (read before the next frame's edit)
    void frameDone(const Engine::World& world, double uploadBytes);
    // The end of the measured frames
    void end(const Engine::World& world);
    // Adds the benchmark's keys to the result object
    void addResult(json11::Json::object& result) const;
    // The readable lines (after WorldBenchmark::summaryText)
    std::string summary() const;
    // Every assertion held
    bool ok() const { return errors.empty(); }
    const std::vector<std::string>& failures() const { return errors; }

   private:
    std::array<double, 4> targets{};
    std::array<int, 8> editsPerBrush{};
    int edits = 0;
    std::vector<double> editMs, encodeMs, visualMs, patched, uploads;
    std::array<uint64_t, 4> noops{};
    std::array<uint64_t, 4> chunksPerOp{}, editsPerOp{};
    uint64_t chunks = 0, samples = 0, encodedBytes = 0, hookStart = 0, hookEnd = 0, masksStart = 0, masksEnd = 0;
    size_t storeBytes = 0;
    std::vector<uint8_t> scratch;
    std::vector<std::string> errors;
    void fail(const std::string& what);
};

}  // namespace Game
}  // namespace Saiga
