/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The staging core of loads (engine-internal; S25, extracted from S11's WorldLoader): the per-type decode of saved
// component blocks into staged values (Fast / Slow / Skip / DropEntity / Replace by the schema match and the
// migrations) and the checks keyed by slot index (classes, dropped entities, hierarchy, PersistentIds, spaces, render
// meshes), over an index space: the slots of a save blob (WorldLoader) or the local entities of an entity group
// (GroupLoader, slots 1..n). One copy of these rules for both.

#pragma once

#include "saiga/engine/ecs/TypeSchema.h"
#include "saiga/engine/save/SaveReader.h"
#include "saiga/engine/save/SchemaTable.h"
#include "saiga/engine/world/World.h"

#include <memory>
#include <set>
#include <string>
#include <vector>

namespace Saiga
{
namespace Engine
{
class StagingCore
{
   public:
    explicit StagingCore(World& w) : w(w), reg(w.registry()) {}

   protected:
    bool fail(const std::string& message)
    {
        if (error.empty()) error = message;
        return false;
    }
    // The values of p.components (blocks into data) into staged values
    bool stageComponents(const std::vector<uint8_t>& data);
    bool mapClasses();
    bool dropEntities();
    bool checkHierarchy();
    // Unique, valid, not above the saved counter (counter: the save's or the world's)
    bool checkPersistent(uint64_t counter);
    // On the staged Transforms: every alive entity's space below saveSpaces, at most one space-state entity per space,
    // a child in its parent's space; transition and entrance targets below the world's space count
    bool checkSpaces();
    // S19b: resolves the staged RenderInstance names (aliases) and sets the cached IDs; an unregistered name is kept
    bool checkRenderMeshes();
    bool claim(ComponentId id, uint32_t index, const std::string& what);
    // The staged value index -> keep; removes the others from a typed stage
    template <class T>
    static void filterStage(StagedOf<T>& s, const std::vector<bool>& keep);

    World& w;
    Registry& reg;
    std::string error;
    ParsedSave p;
    SchemaTable current;
    std::vector<std::unique_ptr<StagedValues>> staged = std::vector<std::unique_ptr<StagedValues>>(kMaxComponents);
    StagedOf<Transform> transforms;
    std::vector<std::vector<bool>> seen = std::vector<std::vector<bool>>(kMaxComponents);
    // The save's space count (1 without SPCE; a group: the world's) and the staged space of each alive index (-1: none)
    size_t saveSpaces = 1;
    std::vector<int64_t> stagedSpace;
    // RenderInstance names that are not registered (kept; logged once after the commit) and their entity count
    std::set<std::string> unknownMeshes;
    size_t unknownMeshEntities = 0;
    std::vector<bool> drop;           // by entity index: dropped by a removed type (DropEntity)
    std::vector<int64_t> generation;  // by entity index after the load: the generation, -1 if not alive
};

template <class T>
void StagingCore::filterStage(StagedOf<T>& s, const std::vector<bool>& keep)
{
    size_t n = 0;
    for (size_t i = 0; i < s.values.size(); ++i)
    {
        if (!keep[i]) continue;
        if (n != i)
        {
            s.values[n]  = std::move(s.values[i]);
            s.indices[n] = s.indices[i];
        }
        n++;
    }
    s.values.resize(n);
    s.indices.resize(n);
}

}  // namespace Engine
}  // namespace Saiga
