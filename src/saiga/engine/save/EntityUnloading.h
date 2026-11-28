/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// EntityUnloading (docs/engine_design.md §9.6, §12.3, §12.4; S25), owned by the World and attached in streaming worlds
// on the host (G18: without streaming nothing unloads): what changed since the last pass (the chunks whose ready state
// flipped to true, whose kept coverage flipped to 0 and that became resident, accumulated from the residency's
// listeners; the created, moved and released roots from the structural log and phase 6's changed list) and the passes
// that act on it:
// - the step-start hook (after S24's refresh, between steps: at once): the space loads, the reloads of chunks that
//   became ready and the chunk records of chunks that became resident (fullReload after a load: every one);
// - phase 7 (commands at the flush, so an aborted phase loses nothing: the accumulated state is consumed by the last
//   command): spaceLoadSystem; chunkEntitySystem: the same reloads and records, then the one unload pass (every
//   unloadable root whose chunk is not kept, in root entity-index order; fullUnload after a load: every root);
//   spaceUnloadSystem (a loaded space without streaming centres and without roots unloads).

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/engine/ecs/Entity.h"
#include "saiga/export.h"

#include <cstdint>
#include <set>
#include <vector>

namespace Saiga
{
namespace Engine
{
class World;

class SAIGA_ENGINE_API EntityUnloading
{
   public:
    explicit EntityUnloading(World& world);
    ~EntityUnloading();
    EntityUnloading(const EntityUnloading&)            = delete;
    EntityUnloading& operator=(const EntityUnloading&) = delete;

    // A streaming host world (the World's constructor): the residency listener, the structural log consumer, the
    // stored-groups part (PartStored) and the step-start hook
    void attach();
    bool attached() const { return consumer >= 0; }
    // The step-start hook
    void stepStart();
    // Phase 7: spaceLoadSystem, chunkEntitySystem, spaceUnloadSystem
    void loadSpaces();
    void chunkPass();
    void unloadSpaces();
    // A committed load or open: the accumulated flips are dropped, the next step start reloads every ready chunk with
    // groups (and records every resident chunk), the next phase 7 examines every root
    void onLoad();
    // Whether a root may be unloaded (not a Player subtree, not owned by a client, no PendingTransition in it, not a
    // state entity, at most kMaxGroupEntities entities)
    bool unloadable(Entity root) const;

   private:
    World& w;
    int listener = -1, consumer = -1, hook = -1;
    std::set<ChunkId> readyChunks, keptLost, residentChunks;
    std::set<uint32_t> candidates, largeRoots;
    bool fullUnload = false, fullReload = false;

    void consumeLog();
    std::vector<ChunkId> reloadList() const;
    std::vector<ChunkId> recordList() const;
    void writeRecord(const ChunkId& chunk);
    bool spaceHasCentre(SpaceId space) const;
    bool spaceHasRoot(SpaceId space) const;
};

}  // namespace Engine
}  // namespace Saiga
