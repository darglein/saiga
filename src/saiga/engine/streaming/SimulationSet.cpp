/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/streaming/SimulationSet.h"

#include "saiga/engine/components/Parent.h"
#include "saiga/engine/components/PendingTransition.h"
#include "saiga/engine/components/Player.h"
#include "saiga/engine/components/StreamingAnchor.h"
#include "saiga/engine/gameplay/Behaviour.h"
#include "saiga/engine/streaming/ChunkResidency.h"
#include "saiga/engine/world/TransformSystem.h"
#include "saiga/engine/world/World.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iterator>

namespace Saiga
{
namespace Engine
{
namespace
{
constexpr uint8_t kStateFlags = Registry::kWorldState | Registry::kSpaceState;
// The cells of a chunk per axis (256 / 32)
constexpr int32_t kCellsPerChunk = 8;
}  // namespace

SimulationSet::SimulationSet(World& world) : w(world) {}

SimulationSet::~SimulationSet()
{
    if (consumer >= 0) w.structuralLog().removeConsumer(consumer);
    if (listener >= 0) w.residency().removeListener(listener);
}

void SimulationSet::attach()
{
    if (consumer >= 0) return;
    consumer = w.structuralLog().addConsumer();
    ResidencyListener l;
    // Host: an alwaysSimulate root's chunk changed its ready state (accumulated until the next pass)
    l.onReadyChanged = [this](const ChunkId& c, bool) { dirtyChunks.insert(c); };
    listener         = w.residency().addListener(std::move(l));
}

const std::vector<std::pair<int32_t, int32_t>>& SimulationSet::disc(double radius) const
{
    auto it = discs.find(radius);
    if (it != discs.end()) return it->second;
    std::vector<std::pair<int32_t, int32_t>> d;
    const double cell = SpatialGrid::kCellSize;
    const int32_t n   = int32_t(std::floor(radius / cell));
    for (int32_t dz = -n; dz <= n; ++dz)
        for (int32_t dx = -n; dx <= n; ++dx)
            if (double(dx * dx + dz * dz) * cell * cell <= radius * radius) d.push_back({dx, dz});
    return discs.emplace(radius, std::move(d)).first->second;
}

void SimulationSet::computeRootSets(std::set<uint32_t>& players, std::set<uint32_t>& anchors) const
{
    Registry& reg = const_cast<Registry&>(w.registry());
    reg.view<const Player>().each([&](Entity e, const Player&) { players.insert(w.rootOf(e).index); });
    reg.view<const StreamingAnchor>().each([&](Entity e, const StreamingAnchor&)
                                           { anchors.insert(w.rootOf(e).index); });
}

std::vector<SimulationSet::CellKey> SimulationSet::computeCentreCells(const std::set<uint32_t>& players,
                                                                      const std::set<uint32_t>& anchors) const
{
    const Registry& reg = w.registry();
    std::vector<CellKey> cells;
    auto add = [&](uint32_t index, bool player)
    {
        const Entity r = reg.aliveAt(index);
        if (!r.valid()) return;
        // A held player is no simulation centre (it is about to leave; G8)
        if (player && w.heldRoot(r)) return;
        const Transform& t = reg.read<Transform>(r);
        if (!t.position().allFinite() || t.space() >= w.spaceCount()) return;
        cells.push_back({t.space(), SpatialGrid::cellOf(t.position().x()), SpatialGrid::cellOf(t.position().z())});
    };
    for (uint32_t i : players) add(i, true);
    for (uint32_t i : anchors)
        if (!players.count(i)) add(i, false);
    std::sort(cells.begin(), cells.end());
    return cells;
}

bool SimulationSet::rule(Entity root, const std::set<uint32_t>& players, const std::set<uint32_t>& anchors,
                         const std::map<CellKey, uint32_t>& cover) const
{
    const Registry& reg = w.registry();
    if (reg.flags(root) & kStateFlags) return true;
    if (w.heldRoot(root)) return false;
    // A client simulates everything but its held roots (G10)
    if (!w.isHost()) return true;
    if (players.count(root.index) || anchors.count(root.index)) return true;
    const Transform& t = reg.read<Transform>(root);
    const Vec3& p      = t.position();
    if (p.allFinite() && cover.count(CellKey{t.space(), SpatialGrid::cellOf(p.x()), SpatialGrid::cellOf(p.z())}))
        return true;
    if (reg.registered(Behaviour::id))
    {
        const Behaviour* b = reg.tryRead<Behaviour>(root);
        if (b && b->alwaysSimulate && p.allFinite() && w.chunkReady(t.space(), p.x(), p.z())) return true;
    }
    return false;
}

bool SimulationSet::bruteForce(Entity root) const
{
    std::set<uint32_t> players, anchors;
    computeRootSets(players, anchors);
    std::map<CellKey, uint32_t> cover;
    for (const CellKey& c : computeCentreCells(players, anchors))
    {
        const double r = w.spaceTemplate(std::get<0>(c)).simulationRadius;
        for (const auto& [dx, dz] : disc(r)) cover[CellKey{std::get<0>(c), std::get<1>(c) + dx, std::get<2>(c) + dz}]++;
    }
    return rule(root, players, anchors, cover);
}

bool SimulationSet::setRoot(Entity root, bool simulated)
{
    Registry& reg       = w.registry();
    const uint8_t flags = reg.flags(root);
    const bool was      = (flags & Registry::kSimulated) != 0;
    if (was == simulated) return false;
    reg.setFlags(root, simulated ? uint8_t(flags | Registry::kSimulated) : uint8_t(flags & ~Registry::kSimulated));
    return true;
}

void SimulationSet::propagateSubtree(Entity root)
{
    if (!w.hasChildren(root)) return;
    Registry& reg  = w.registry();
    const bool sim = (reg.flags(root) & Registry::kSimulated) != 0;
    for (Entity x : w.subtree(root))
        if (x != root) setRoot(x, sim);
}

void SimulationSet::propagateAll()
{
    ensureHierarchyOrder(w);
    Registry& reg = w.registry();
    for (Entity child : w.hierarchyOrder)
    {
        if (!reg.alive(child)) continue;
        const Parent* p = reg.tryRead<Parent>(child);
        if (!p || !reg.alive(p->entity)) continue;
        setRoot(child, (reg.flags(p->entity) & Registry::kSimulated) != 0);
    }
}

void SimulationSet::consumeLog(std::vector<Entity>* now)
{
    if (consumer < 0) return;
    const Registry& reg = w.registry();
    w.structuralLog().consume(consumer,
                              [&](const StructuralLog::Record& r)
                              {
                                  const ComponentId c = r.component;
                                  if (c == Parent::id) parentsChanged = true;
                                  const bool created = c == Transform::id && r.added;
                                  const bool held = c == PendingTransition::id || c == Parent::id || c == Player::id ||
                                                    c == StreamingAnchor::id;
                                  if (!created && !held && c != Behaviour::id) return;
                                  if (!reg.alive(r.entity)) return;
                                  if (now && (created || held))
                                      now->push_back(r.entity);
                                  else
                                      dirty.insert(r.entity.index);
                              });
}

void SimulationSet::evaluateCreated()
{
    if (consumer < 0) return;
    std::vector<Entity> now;
    consumeLog(&now);
    if (!init || now.empty()) return;
    // The roots of the created entities and the structural changes now (with the coverage of the last pass); their
    // other effects (a new centre's coverage) follow in the next pass
    playerRoots.clear();
    anchorRoots.clear();
    computeRootSets(playerRoots, anchorRoots);
    std::set<uint32_t> roots;
    for (Entity e : now)
    {
        if (!w.alive(e)) continue;
        roots.insert(w.rootOf(e).index);
        dirty.insert(e.index);
    }
    for (uint32_t i : roots)
    {
        const Entity r = w.registry().aliveAt(i);
        if (!r.valid()) continue;
        evaluate(r);
        propagateSubtree(r);
    }
}

void SimulationSet::refresh(Entity e)
{
    if (consumer < 0 || !init || !w.alive(e)) return;
    playerRoots.clear();
    anchorRoots.clear();
    computeRootSets(playerRoots, anchorRoots);
    const Entity r = w.rootOf(e);
    evaluate(r);
    propagateSubtree(r);
    // Its effect on the coverage (a moved centre) in the next pass
    dirty.insert(r.index);
}

void SimulationSet::invalidate()
{
    init = false;
    centreCells.clear();
    coverage.clear();
    dirty.clear();
    dirtyChunks.clear();
    clientFrozen.clear();
    parentsChanged = false;
}

void SimulationSet::run(bool full)
{
    Registry& reg = w.registry();
    playerRoots.clear();
    anchorRoots.clear();
    computeRootSets(playerRoots, anchorRoots);
    std::set<uint32_t> roots;
    auto addRoot = [&](uint32_t index)
    {
        const Entity e = reg.aliveAt(index);
        if (e.valid()) roots.insert(w.rootOf(e).index);
    };
    if (w.isHost())
    {
        // The coverage: the centre cells' difference to the last pass; the cells whose coverage flipped
        std::vector<CellKey> cells = computeCentreCells(playerRoots, anchorRoots);
        std::set<CellKey> flipped;
        if (full)
        {
            coverage.clear();
            centreCells.clear();
        }
        std::vector<CellKey> removed, added;
        std::set_difference(centreCells.begin(), centreCells.end(), cells.begin(), cells.end(),
                            std::back_inserter(removed));
        std::set_difference(cells.begin(), cells.end(), centreCells.begin(), centreCells.end(),
                            std::back_inserter(added));
        for (const CellKey& c : removed)
            for (const auto& [dx, dz] : disc(w.spaceTemplate(std::get<0>(c)).simulationRadius))
            {
                const CellKey k{std::get<0>(c), std::get<1>(c) + dx, std::get<2>(c) + dz};
                auto it = coverage.find(k);
                if (it == coverage.end()) continue;
                if (--it->second == 0)
                {
                    coverage.erase(it);
                    flipped.insert(k);
                }
            }
        for (const CellKey& c : added)
            for (const auto& [dx, dz] : disc(w.spaceTemplate(std::get<0>(c)).simulationRadius))
            {
                const CellKey k{std::get<0>(c), std::get<1>(c) + dx, std::get<2>(c) + dz};
                if (++coverage[k] == 1)
                {
                    // Flipped back within this pass: no net change
                    if (!flipped.erase(k)) flipped.insert(k);
                }
            }
        centreCells = std::move(cells);
        if (!full)
        {
            const SpatialGrid& grid = w.spatialGrid();
            for (const CellKey& k : flipped)
                grid.forEachInCell(std::get<0>(k), std::get<1>(k), std::get<2>(k),
                                   [&](const SpatialGrid::Item& it) { addRoot(it.entity.index); });
            // A chunk's ready state matters only to alwaysSimulate roots
            const bool behaviours = reg.registered(Behaviour::id);
            for (const ChunkId& c : dirtyChunks)
                for (int32_t z = 0; z < kCellsPerChunk && behaviours; ++z)
                    for (int32_t x = 0; x < kCellsPerChunk; ++x)
                        grid.forEachInCell(c.space, c.x * kCellsPerChunk + x, c.z * kCellsPerChunk + z,
                                           [&](const SpatialGrid::Item& it)
                                           {
                                               if (!reg.alive(it.entity)) return;
                                               const Entity r     = w.rootOf(it.entity);
                                               const Behaviour* b = reg.tryRead<Behaviour>(r);
                                               if (b && b->alwaysSimulate) roots.insert(r.index);
                                           });
            // The centres themselves (a new or moved player root)
            for (uint32_t i : playerRoots) addRoot(i);
            for (uint32_t i : anchorRoots) addRoot(i);
        }
    }
    else
    {
        // Client: the held roots and those frozen until now
        for (uint32_t i : clientFrozen) addRoot(i);
        reg.view<const PendingTransition>().each([&](Entity e, const PendingTransition&) { addRoot(e.index); });
    }
    dirtyChunks.clear();
    for (uint32_t i : dirty) addRoot(i);
    dirty.clear();
    // The moved roots whose cell changed since their last evaluation (a move inside a cell changes nothing: coverage
    // flips, ready flips and structural changes have their own triggers); a moved child follows its root
    for (uint32_t i : moved)
    {
        const Entity e = reg.aliveAt(i);
        if (!e.valid() || reg.has<Parent>(e)) continue;
        const Transform& t = reg.read<Transform>(e);
        const int32_t cx = SpatialGrid::cellOf(t.position().x()), cz = SpatialGrid::cellOf(t.position().z());
        if (i < lastCell.size())
        {
            const EvaluatedCell& c = lastCell[i];
            if (c.valid && c.space == t.space() && c.x == cx && c.z == cz) continue;
        }
        roots.insert(i);
    }
    moved.clear();
    bool flippedAny = false;
    if (full)
    {
        // Every alive root (the state entities too); then every child takes its parent's flag
        reg.view<const Transform>().each(
            [&](Entity e, const Transform&)
            {
                if (!reg.has<Parent>(e)) evaluate(e);
            });
        propagateAll();
        init           = true;
        parentsChanged = false;
    }
    else
    {
        for (uint32_t i : roots)
        {
            const Entity r = reg.aliveAt(i);
            if (!r.valid()) continue;
            flippedAny = evaluate(r) || flippedAny;
        }
        if (flippedAny || parentsChanged) propagateAll();
        parentsChanged = false;
    }
    if (!w.isHost())
    {
        clientFrozen.clear();
        for (uint32_t i : roots)
        {
            const Entity r = reg.aliveAt(i);
            if (r.valid() && !(reg.flags(r) & Registry::kSimulated)) clientFrozen.insert(i);
        }
    }
}

bool SimulationSet::evaluate(Entity root)
{
    const bool changed = setRoot(root, rule(root, playerRoots, anchorRoots, coverage));
    const Transform& t = w.registry().read<Transform>(root);
    if (root.index >= lastCell.size()) lastCell.resize(size_t(root.index) + 1);
    EvaluatedCell& c = lastCell[root.index];
    c.space          = t.space();
    c.x              = SpatialGrid::cellOf(t.position().x());
    c.z              = SpatialGrid::cellOf(t.position().z());
    c.valid          = true;
    w.residency().statsForUpdate().evaluations++;
    return changed;
}

int SimulationSet::countSimulated()
{
    const Registry& reg = w.registry();
    int n               = 0;
    for (uint32_t i = 1; i < reg.slotCount(); ++i)
    {
        const Entity e = reg.aliveAt(i);
        n += e.valid() && (reg.flags(e) & Registry::kSimulated) ? 1 : 0;
    }
    w.residency().statsForUpdate().simulated = n;
    return n;
}

void SimulationSet::update()
{
    const auto start = std::chrono::steady_clock::now();
    consumeLog(nullptr);
    if (!init)
        run(true);
    else
    {
        // The moved entities (this or the previous step, phase 6's list)
        moved = w.changedTransforms();
        run(false);
    }
    // The count is a statistic: a pass over the slots every 16th update (the tools count on demand)
    if (updates++ % 16 == 0) countSimulated();
    w.residency().statsForUpdate().lastSimulationMs =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
}

void SimulationSet::stepStart()
{
    consumeLog(nullptr);
    if (!init)
    {
        run(true);
        countSimulated();
        return;
    }
    // The transforms changed between steps (the flags shifted at the end of the last step)
    moved.clear();
    w.registry().collectChangedThisStep(moved);
    run(false);
}

}  // namespace Engine
}  // namespace Saiga
