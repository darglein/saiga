/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// spatialGridSystem (phase 6): created and destroyed entities from the structural log, moved ones from the changed
// list.

#include "saiga/engine/world/SpatialGridSystem.h"

#include "saiga/engine/world/World.h"

namespace Saiga
{
namespace Engine
{
void spatialGridSystem(World& w, double)
{
    SpatialGrid& grid   = w.spatialGrid();
    const Registry& reg = w.registry();
    // Structural changes since the last update, in order (a slot reused after a destroy leaves, then enters again)
    w.structuralLog().consume(w.spatialGridConsumer(),
                              [&](const StructuralLog::Record& r)
                              {
                                  if (r.component != Transform::id) return;
                                  if (!r.added)
                                  {
                                      if (grid.entityAt(r.entity.index) == r.entity) grid.remove(r.entity.index);
                                  }
                                  else if (reg.alive(r.entity))
                                  {
                                      const Transform& t = reg.read<Transform>(r.entity);
                                      grid.update(r.entity, t.space(), t.position());
                                  }
                              });
    // Moved entities (changed in this or the previous step: phases 7 and 8 of the last step included)
    for (uint32_t index : w.changedTransforms())
    {
        const Entity e = grid.entityAt(index);
        if (!reg.alive(e)) continue;
        const Transform& t = reg.read<Transform>(e);
        grid.update(e, t.space(), t.position());
    }
}

void rebuildSpatialGrid(World& w)
{
    SpatialGrid& grid = w.spatialGrid();
    grid.clear();
    // The log records up to now are reflected by the rebuild
    w.structuralLog().consume(w.spatialGridConsumer(), [](const StructuralLog::Record&) {});
    w.viewAll<const Transform>().each([&](Entity e, const Transform& t) { grid.update(e, t.space(), t.position()); });
}

}  // namespace Engine
}  // namespace Saiga
