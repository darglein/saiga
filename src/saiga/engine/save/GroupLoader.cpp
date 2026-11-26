/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/save/GroupLoader.h"

#include "saiga/engine/components/Parent.h"
#include "saiga/engine/components/Persistent.h"
#include "saiga/engine/gameplay/Behaviour.h"
#include "saiga/engine/net/NetOwner.h"

namespace Saiga
{
namespace Engine
{
bool GroupLoader::stage(const StoredGroup& group, const SchemaTable& schema, const SchemaTable& currentTable,
                        const ChunkId& c, bool spaceState)
{
    chunk      = c;
    spaceGroup = spaceState;
    if (!group.bytes) return fail("the group's bytes are not in memory");
    const std::vector<uint8_t>& bytes = *group.bytes;
    if (!parseGroupForm(bytes.data(), bytes.size(), schema.types.size(), form, error)) return false;
    // The group as a save of its own: slots 1..n alive (generation 0: the local handles), the blocks into its bytes
    p        = ParsedSave();
    p.schema = schema;
    p.slots.resize(size_t(form.entities) + 1);
    for (size_t i = 1; i < p.slots.size(); ++i)
    {
        p.slots[i].generation = 0;
        p.slots[i].state      = uint8_t(SlotState::Alive);
        p.slots[i].flags      = (spaceState && i == 1) ? Registry::kSpaceState : uint8_t(0);
    }
    if (spaceState && form.entities != 1) return fail("a space record's group has more than one entity");
    p.components        = form.blocks;
    p.persistentCounter = w.persistentCounter;
    current             = currentTable;
    saveSpaces          = w.spaceCount();
    drop.assign(p.slots.size(), false);
    if (!stageComponents(bytes) || !mapClasses() || !dropEntities() || !checkHierarchy() || !checkSpaces() ||
        !checkPersistent(w.persistentCounter) || !checkRenderMeshes())
        return false;
    // The group's own rules
    for (size_t k = 0; k < transforms.indices.size(); ++k)
    {
        const uint32_t i = transforms.indices[k];
        if (generation[i] < 0) continue;
        if (transforms.values[k].space() != chunk.space)
            return fail("local entity " + std::to_string(i - 1) + " is in space " +
                        std::to_string(transforms.values[k].space()) + ", the chunk in space " +
                        std::to_string(chunk.space));
    }
    if (staged[Parent::id])
    {
        const auto& sp = static_cast<StagedOf<Parent>&>(*staged[Parent::id]);
        for (size_t k = 0; k < sp.values.size(); ++k)
        {
            const uint32_t child = sp.indices[k];
            if (generation[child] < 0) continue;
            if (child == 1) return fail("the root has a Parent");
            if (sp.values[k].entity.index >= child)
                return fail("local entity " + std::to_string(child - 1) +
                            " has a Parent that is not an earlier entity");
        }
    }
    // Every non-root entity is a descendant of the root (it has a Parent)
    for (size_t i = 2; i < p.slots.size(); ++i)
    {
        if (generation[i] < 0) continue;
        const auto& s = seen[Parent::id];
        if (s.empty() || !s[i]) return fail("local entity " + std::to_string(i - 1) + " has no Parent");
    }
    if (staged[Persistent::id])
    {
        const auto& sp = static_cast<StagedOf<Persistent>&>(*staged[Persistent::id]);
        std::vector<uint64_t> ids;
        for (size_t k = 0; k < sp.values.size(); ++k)
        {
            const uint64_t v = sp.values[k].persistentId.value;
            if (generation[sp.indices[k]] >= 0 && w.resolve(PersistentId{v}).valid())
                return fail("the PersistentId " + std::to_string(v) + " is in use");
            ids.push_back(v);
        }
        std::sort(ids.begin(), ids.end());
        if (ids != form.persistentIds) return fail("the PersistentId list does not match the components");
    }
    else if (!form.persistentIds.empty())
        return fail("the PersistentId list does not match the components");
    return true;
}

void GroupLoader::commit(std::vector<std::pair<Entity, double>>& objects)
{
    Registry& r = w.reg;
    created.assign(p.slots.size(), Entity{});
    createdEntities = 0;
    for (size_t i = 1; i < p.slots.size(); ++i)
    {
        if (generation[i] < 0) continue;
        auto record = w.log.prepare(1);
        created[i]  = r.create();
        record.append(created[i], Transform::id, true);
        if (p.slots[i].flags) r.setFlags(created[i], uint8_t(r.flags(created[i]) | p.slots[i].flags));
        createdEntities++;
    }
    const EntityRemap remap = [&](Entity e) -> Entity
    {
        if (e.generation != 0 || e.index == 0 || e.index >= created.size()) return Entity{};
        return created[e.index];
    };
    for (size_t k = 0; k < transforms.indices.size(); ++k)
    {
        const Entity e = created[transforms.indices[k]];
        if (e.valid()) w.setLoadedTransform(e, transforms.values[k]);
    }
    const double dt      = w.dt();
    const uint64_t steps = w.stepIndex();
    const double elapsed = steps > form.unloadStep ? double(steps - form.unloadStep) * dt : 0.0;
    for (size_t id = 1; id < kMaxComponents; ++id)
    {
        if (!staged[id]) continue;
        StagedValues& stage = *staged[id];
        r.poolById(ComponentId(id))->remapStaged(stage, remap);
        for (size_t k = 0; k < stage.indices.size(); ++k)
        {
            const Entity e = created[stage.indices[k]];
            if (!e.valid()) continue;
            // A Parent whose target was dropped (a removed type) cannot happen: dropEntities drops the descendants
            Entity parent;
            if (id == Parent::id)
            {
                parent = static_cast<StagedOf<Parent>&>(stage).values[k].entity;
                w.reserveChildSlot(parent);
            }
            uint64_t pid = 0;
            if (id == Persistent::id)
            {
                pid = static_cast<StagedOf<Persistent>&>(stage).values[k].persistentId.value;
                w.persistentIndex.emplace(pid, e);
            }
            auto record = w.log.prepare(1);
            r.addStagedAt(e, ComponentId(id), stage, k);
            record.append(e, ComponentId(id), true);
            if (id == Parent::id)
            {
                w.changeChildCount(parent, +1);
                w.hierarchyDirty  = true;
                w.ownerCacheDirty = true;
            }
            if (id == NetOwner::id) w.ownerCacheDirty = true;
            if (id == Behaviour::id) objects.push_back({e, elapsed});
        }
    }
}

}  // namespace Engine
}  // namespace Saiga
