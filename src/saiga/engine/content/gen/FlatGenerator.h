/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// FlatGenerator (docs/engine_design.md §12.1; S23a): the engine's generator 0, used by every template without a
// generator. Height 0, no slots, an empty plan and an all-zero grass mask; legacyTerrain (S20's flat default: no
// generation state) and no terrain (D1: nothing drawn or streamed).

#pragma once

#include "saiga/engine/content/Generator.h"
#include "saiga/export.h"

namespace Saiga
{
namespace Engine
{
class SAIGA_ENGINE_API FlatGenerator final : public Generator
{
   public:
    const char* name() const override { return "flat"; }
    uint16_t version() const override { return 1; }
    std::vector<GeneratorSlot> slots() const override { return {}; }
    bool legacyTerrain() const override { return true; }
    bool hasTerrain() const override { return false; }
    void plan(uint16_t, WorldPlan&) const override {}
    int32_t height(uint16_t, const WorldPlan&, int64_t, int64_t) const override { return 0; }
    bool candidate(uint16_t, const WorldPlan&, ChunkId, uint32_t, uint32_t, Candidate&) const override { return false; }
    uint32_t candidatesPerChunk(uint32_t) const override { return 0; }
    uint32_t keptCount(uint32_t, int) const override { return 0; }
    void grassMask(uint16_t version, const WorldPlan& plan, ChunkId chunk, GrassMask& out) const override;
};

// An all-zero mask of the template's size (256 / baseSpacing per side)
SAIGA_ENGINE_API void zeroGrassMask(const WorldPlan& plan, GrassMask& out);

}  // namespace Engine
}  // namespace Saiga
