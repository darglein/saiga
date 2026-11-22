/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// HookGenerator (docs/engine_design.md §7.9, §12.1; S23a): adapts an S20 height hook (TerrainHeightHook, called with
// plan.seed) to the Generator interface. Beside the terrain code, not content code (not linted): the float prototype
// fixture of the games (name "prototype", integerOnly false) and the engine tests' hooks. height() clamps the hook's
// result (clampTerrainHeight, E7); no slots, no sites, legacyTerrain (no generation state), with terrain. grassMask
// evaluates the hook at the chunk's samples and applies S20's grass rule with the given thresholds (all zero without a
// rule). Heights, tests and goldens of the hook users are unchanged.

#pragma once

#include "saiga/core/math/TerrainSurface.h"
#include "saiga/engine/content/Generator.h"
#include "saiga/export.h"

#include <optional>
#include <string>

namespace Saiga
{
namespace Engine
{
// S20's grass rule (Vulkan::prototypeGrassMask): the thresholds of TerrainViewSettings
struct HookGrassRule
{
    double minNormalY = 0.82, minAboveSea = 0.4;
};

class SAIGA_ENGINE_API HookGenerator final : public Generator
{
   public:
    HookGenerator(std::string name, TerrainHeightHook hook, std::optional<HookGrassRule> grass = std::nullopt,
                  bool integerOnly = true);
    const char* name() const override { return generatorName.c_str(); }
    uint16_t version() const override { return 1; }
    std::vector<GeneratorSlot> slots() const override { return {}; }
    bool integerOnly() const override { return integer; }
    bool legacyTerrain() const override { return true; }
    bool hasTerrain() const override { return true; }
    void plan(uint16_t, WorldPlan&) const override {}
    int32_t height(uint16_t version, const WorldPlan& plan, int64_t x, int64_t z) const override;
    bool candidate(uint16_t, const WorldPlan&, ChunkId, uint32_t, uint32_t, Candidate&) const override { return false; }
    uint32_t candidatesPerChunk(uint32_t) const override { return 0; }
    uint32_t keptCount(uint32_t, int) const override { return 0; }
    void grassMask(uint16_t version, const WorldPlan& plan, ChunkId chunk, GrassMask& out) const override;

    // The rule in Q16 / height units (rounded; the terrain uses grassRule()'s exact thresholds for legacy spaces)
    GrassEditRule grassEditRule() const override;
    TerrainHeightHook heightHook() const { return hook; }
    // The constructor's thresholds (S20's mask of a legacy space; none: all zero)
    const std::optional<HookGrassRule>& grassRule() const { return grass; }

   private:
    std::string generatorName;
    TerrainHeightHook hook = nullptr;
    std::optional<HookGrassRule> grass;
    bool integer = true;
};

}  // namespace Engine
}  // namespace Saiga
