/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// AuthoredSpaceGenerator (docs/engine_design.md §12.1, §12.4; S23a): an enclosed space that is prefabs only (a
// dungeon). Its plan holds one site at the centre of the middle chunk (S21's site origin, E4: (sizeChunks / 2) * 256
// + 128 m in x and z, baseHeight 0) whose footprint is the whole space; no slots, height 0, no terrain. Not legacy: a
// world with it writes generation state.

#pragma once

#include "saiga/engine/content/Generator.h"
#include "saiga/export.h"

#include <string>

namespace Saiga
{
namespace Engine
{
class SAIGA_ENGINE_API AuthoredSpaceGenerator final : public Generator
{
   public:
    AuthoredSpaceGenerator(std::string name, std::string prefab, uint16_t prefabVersion = 1);
    const char* name() const override { return generatorName.c_str(); }
    uint16_t version() const override { return 1; }
    std::vector<GeneratorSlot> slots() const override { return {}; }
    bool hasTerrain() const override { return false; }
    void plan(uint16_t version, WorldPlan& plan) const override;
    int32_t height(uint16_t, const WorldPlan&, int64_t, int64_t) const override { return 0; }
    bool candidate(uint16_t, const WorldPlan&, ChunkId, uint32_t, uint32_t, Candidate&) const override { return false; }
    uint32_t candidatesPerChunk(uint32_t) const override { return 0; }
    uint32_t keptCount(uint32_t, int) const override { return 0; }
    void grassMask(uint16_t version, const WorldPlan& plan, ChunkId chunk, GrassMask& out) const override;

   private:
    std::string generatorName, prefab;
    uint16_t prefabVersion = 1;
};

}  // namespace Engine
}  // namespace Saiga
