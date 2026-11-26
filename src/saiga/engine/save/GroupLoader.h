/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// GroupLoader (docs/engine_design.md §9.6, §15.2; S25): validate-then-commit of one stored entity group on the staging
// core of loads. stage() decodes the group against its schema table and the current one (Same types from the binary
// form, Compatible / Migrate through the field list and the migrations: lazy migration; Removed policies) and checks
// it: local indices in range, Parent only to an earlier local entity and the root without one, known classes, every
// Transform::space the chunk's, transition and entrance targets in range, PersistentIds valid and not in use, render
// meshes (unknown names kept). commit() creates the entities at free slots (new handles), adds the components in block
// order with their log records and the bookkeeping (children, Persistent index, owners), maps every Entity field
// Entity{local + 1, 0} to the new handle and returns the game objects for World::startReloadedObjects.

#pragma once

#include "saiga/engine/save/ChunkStore.h"
#include "saiga/engine/save/EntityGroup.h"
#include "saiga/engine/save/StagingCore.h"

#include <utility>
#include <vector>

namespace Saiga
{
namespace Engine
{
class GroupLoader : public StagingCore
{
   public:
    explicit GroupLoader(World& w) : StagingCore(w) {}
    // false with errorMessage(); nothing changed. spaceState: the group of a space record (local entity 1 is the
    // space-state entity of chunk.space)
    bool stage(const StoredGroup& group, const SchemaTable& schema, const SchemaTable& currentTable,
               const ChunkId& chunk, bool spaceState);
    // Creates the staged entities; objects gets the entities with a Behaviour and their elapsed time
    void commit(std::vector<std::pair<Entity, double>>& objects);
    const std::string& errorMessage() const { return error; }
    // The created root (Entity{} if it was dropped by a removed type)
    Entity root() const { return created.size() > 1 ? created[1] : Entity{}; }
    uint32_t createdCount() const { return createdEntities; }

   private:
    GroupForm form;
    ChunkId chunk;
    bool spaceGroup = false;
    std::vector<Entity> created;
    uint32_t createdEntities = 0;
};

}  // namespace Engine
}  // namespace Saiga
