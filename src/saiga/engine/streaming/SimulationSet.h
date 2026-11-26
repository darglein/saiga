/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// SimulationSet (docs/engine_design.md §12.3; S24, G8), owned by the World: the Simulated tag (Registry::kSimulated)
// of a streaming world, exact and incremental by the spatial grid's 32 m cells. Per space the coverage of the cells
// by the simulation discs of the centre cells (player roots that are not held, StreamingAnchor roots; a cell is
// covered by a centre cell when (dx^2 + dz^2) * 32^2 <= simulationRadius^2), and a root rule, in order: world-state or
// space-state entity: simulated; a held player root: frozen; a player root or a root with a StreamingAnchor in its
// subtree: simulated; its cell covered: simulated; Behaviour::alwaysSimulate and its chunk ready: simulated; else
// frozen. Descendants take their root's flag. A client: every entity simulated except the held player roots (G10).
//
// Evaluated: in phase 7 (simulationSystem; the roots of the cells whose coverage flipped, the moved entities, the
// entities of the chunks whose ready state flipped and the structural changes of Player, StreamingAnchor, Parent,
// PendingTransition and Behaviour), at the step start (the changes made between steps), after every command flush
// (created entities, held toggles) and at once by refreshSimulated. A full evaluation while not initialised (the first
// step, a load). Derived data: not saved, not hashed.

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/engine/ecs/Entity.h"
#include "saiga/export.h"

#include <array>
#include <cstdint>
#include <map>
#include <set>
#include <tuple>
#include <vector>

namespace Saiga
{
namespace Engine
{
class World;

class SAIGA_ENGINE_API SimulationSet
{
   public:
    explicit SimulationSet(World& world);
    ~SimulationSet();
    SimulationSet(const SimulationSet&)            = delete;
    SimulationSet& operator=(const SimulationSet&) = delete;

    // The World's constructor of a streaming world: the structural log consumer and the residency listener
    void attach();
    // Phase 7 (simulationSystem, both roles)
    void update();
    // The step start: the changes made between steps (the transforms changed since the last step, the log)
    void stepStart();
    // After a command flush (and between steps): the created entities and the held toggles now
    void evaluateCreated();
    // Re-evaluates the root of e (and its subtree) now
    void refresh(Entity e);
    // A load: the next evaluation is a full one (the World runs it before tryLoad returns)
    void invalidate();
    bool initialised() const { return init; }
    // Counts the simulated entities into StreamingStats::simulated (a pass over the slots: every 16th phase-7 update,
    // and on demand by the tools) and returns it
    int countSimulated();
    // The rule for one root from scratch (tests: the brute-force comparison; the coverage from the current positions)
    bool bruteForce(Entity root) const;

   private:
    using CellKey = std::tuple<SpaceId, int32_t, int32_t>;
    World& w;
    int consumer = -1, listener = -1;
    bool init = false;
    std::vector<CellKey> centreCells;  // sorted, with duplicates
    std::map<CellKey, uint32_t> coverage;
    std::set<uint32_t> dirty;  // entity indices
    // The moved entities of this pass (a root is evaluated only when its cell changed since its last evaluation: the
    // rule depends on the position only through the cell and its chunk)
    std::vector<uint32_t> moved;
    struct EvaluatedCell
    {
        SpaceId space = 0;
        int32_t x = 0, z = 0;
        bool finite = true;
        bool valid  = false;
    };
    std::vector<EvaluatedCell> lastCell;  // by entity index
    uint64_t updates = 0;
    std::set<ChunkId> dirtyChunks;
    bool parentsChanged = false;
    // The roots with a Player / StreamingAnchor in their subtree (by index), computed per pass
    std::set<uint32_t> playerRoots, anchorRoots;
    // Client: the roots frozen now (held), re-evaluated every pass
    std::set<uint32_t> clientFrozen;
    // The disc offsets per simulation radius
    mutable std::map<double, std::vector<std::pair<int32_t, int32_t>>> discs;

    const std::vector<std::pair<int32_t, int32_t>>& disc(double radius) const;
    void computeRootSets(std::set<uint32_t>& players, std::set<uint32_t>& anchors) const;
    std::vector<CellKey> computeCentreCells(const std::set<uint32_t>& players, const std::set<uint32_t>& anchors) const;
    // The rule (G8) with the given sets and coverage
    bool rule(Entity root, const std::set<uint32_t>& players, const std::set<uint32_t>& anchors,
              const std::map<CellKey, uint32_t>& cover) const;
    // Consumes the structural log: created entities and held toggles to `now`, the rest to dirty
    void consumeLog(std::vector<Entity>* now);
    // The pass: centres, coverage flips, dirty chunks, dirty roots, propagation (full: every root)
    void run(bool full);
    // Sets the flag of root (and returns whether it changed)
    bool setRoot(Entity root, bool simulated);
    // The subtree of root takes its flag (collectSubtree; few roots) / every child takes its parent's (hierarchyOrder)
    void propagateSubtree(Entity root);
    void propagateAll();
    // Evaluates a root and records its cell; returns whether its flag changed
    bool evaluate(Entity root);
};

}  // namespace Engine
}  // namespace Saiga
