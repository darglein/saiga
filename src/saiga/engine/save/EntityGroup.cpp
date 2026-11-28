/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/save/EntityGroup.h"

#include "saiga/engine/components/Persistent.h"
#include "saiga/engine/gameplay/Behaviour.h"
#include "saiga/engine/world/World.h"

#include <algorithm>

#include <unordered_map>

namespace Saiga
{
namespace Engine
{
bool parseGroupForm(const uint8_t* data, size_t size, size_t schemaTypes, GroupForm& out, std::string& error)
{
    out = GroupForm();
    BinaryReader r(data, size);
    auto fail = [&](const std::string& m)
    {
        error = "entity group: " + m;
        return false;
    };
    uint32_t pids = 0, blocks = 0;
    r.read(out.unloadStep);
    r.read(out.entities);
    r.read(pids);
    if (!r.ok()) return fail(r.error());
    if (out.entities == 0 || out.entities > kMaxGroupEntities)
        return fail("the entity count " + std::to_string(out.entities) + " is not in 1.." +
                    std::to_string(kMaxGroupEntities));
    if (pids > out.entities || uint64_t(pids) * 8 > r.remaining()) return fail("the PersistentId count does not fit");
    out.persistentIds.resize(pids);
    for (uint32_t i = 0; i < pids; ++i)
    {
        r.read(out.persistentIds[i]);
        if (out.persistentIds[i] == 0) return fail("an invalid PersistentId");
        if (i > 0 && out.persistentIds[i] <= out.persistentIds[i - 1])
            return fail("the PersistentIds are not ascending and unique");
    }
    r.read(blocks);
    if (!r.ok()) return fail(r.error());
    if (blocks > schemaTypes || uint64_t(blocks) * 16 > r.remaining()) return fail("the block count does not fit");
    std::vector<bool> seenType(schemaTypes, false);
    for (uint32_t b = 0; b < blocks; ++b)
    {
        ParsedSave::Block block;
        uint32_t count  = 0;
        uint64_t length = 0;
        r.read(block.schemaIndex);
        r.read(count);
        r.read(length);
        if (!r.ok()) return fail(r.error());
        if (block.schemaIndex >= schemaTypes)
            return fail("the schema index " + std::to_string(block.schemaIndex) + " is out of range");
        if (seenType[block.schemaIndex]) return fail("a component type appears twice");
        seenType[block.schemaIndex] = true;
        if (length > r.remaining()) return fail("a block does not fit");
        if (count > out.entities || uint64_t(count) * 4 > length) return fail("a block's value count does not fit");
        block.count = count;
        block.begin = r.position();
        block.end   = r.position() + size_t(length);
        r.skip(length);
        out.blocks.push_back(block);
    }
    if (!r.atEnd()) return fail("bytes left over");
    return true;
}

void writeGroup(const World& world, Entity root, const SchemaTable& schema, uint64_t step, GroupWriteResult& out)
{
    const Registry& reg = world.registry();
    out.bytes.clear();
    out.entities          = world.subtree(root);
    out.droppedReferences = 0;
    // Entity index -> local index + 1
    std::unordered_map<uint32_t, uint32_t> local;
    for (size_t i = 0; i < out.entities.size(); ++i) local[out.entities[i].index] = uint32_t(i + 1);
    const EntityRemap remap = [&](Entity e) -> Entity
    {
        if (!e.valid()) return Entity{};
        auto it = local.find(e.index);
        if (it != local.end() && out.entities[it->second - 1] == e) return Entity{it->second, 0};
        out.droppedReferences++;
        return Entity{};
    };
    // The callers never unload a larger subtree (EntityUnloading::unloadable): parseGroupForm would refuse it
    SAIGA_ASSERT(out.entities.size() <= kMaxGroupEntities, "writeGroup: the subtree exceeds kMaxGroupEntities");
    BinaryWriter w(out.bytes);
    w.write(step);
    w.write(uint32_t(out.entities.size()));
    std::vector<uint64_t> pids;
    if (reg.registered(Persistent::id))
        for (Entity e : out.entities)
            if (const Persistent* p = reg.tryRead<Persistent>(e)) pids.push_back(p->persistentId.value);
    std::sort(pids.begin(), pids.end());
    w.write(uint32_t(pids.size()));
    for (uint64_t p : pids) w.write(p);
    auto schemaIndex = [&](const char* name)
    {
        const SchemaType* t = schema.find(SchemaKind::Component, name);
        SAIGA_ASSERT(t, std::string("entity group: ") + name + " is not in the schema");
        return uint32_t(t - schema.types.data());
    };
    const size_t blockCountAt = out.bytes.size();
    w.write(uint32_t(0));
    uint32_t blocks = 0;
    for (size_t id = 0; id < kMaxComponents; ++id)
    {
        if (!reg.savedComponent(ComponentId(id))) continue;
        const PoolBase* pool = id == Transform::id ? nullptr : reg.poolById(ComponentId(id));
        uint32_t count       = 0;
        for (Entity e : out.entities) count += reg.entityMask(e).test(ComponentId(id)) ? 1 : 0;
        if (count == 0) continue;
        blocks++;
        w.write(schemaIndex(reg.componentInfo(ComponentId(id)).name));
        w.write(count);
        const size_t lengthAt = out.bytes.size();
        w.write(uint64_t(0));
        const size_t start = out.bytes.size();
        for (size_t i = 0; i < out.entities.size(); ++i)
        {
            const Entity e = out.entities[i];
            if (!reg.entityMask(e).test(ComponentId(id))) continue;
            w.write(uint32_t(i + 1));
            if (id == Transform::id)
                w.write(reg.transformAt(e.index));
            else
                pool->appendRemapped(e.index, out.bytes, remap);
        }
        patchU64(out.bytes, lengthAt, uint64_t(out.bytes.size() - start));
    }
    for (int i = 0; i < 4; ++i) out.bytes[blockCountAt + size_t(i)] = uint8_t(blocks >> (8 * i));
    SAIGA_ASSERT(w.ok(), "entity group: " + w.error());
}

bool describeGroup(const std::vector<uint8_t>& bytes, const SchemaTable& schema, std::string& rootClass,
                   uint32_t& entities)
{
    GroupForm f;
    std::string error;
    if (!parseGroupForm(bytes.data(), bytes.size(), schema.types.size(), f, error)) return false;
    entities  = f.entities;
    rootClass = "";
    for (const ParsedSave::Block& b : f.blocks)
    {
        const SchemaType& t = schema.types[b.schemaIndex];
        if (t.kind != SchemaKind::Component || t.name != Behaviour::name || b.count == 0) continue;
        // The first value: local index, then the class ID (the first field of every Behaviour version)
        BinaryReader r(bytes.data() + b.begin, b.end - b.begin);
        uint32_t localIndex = 0;
        ClassId classId     = 0;
        r.read(localIndex);
        r.read(classId);
        if (r.ok() && localIndex == 1)
            if (const std::string* n = schema.className(classId)) rootClass = *n;
    }
    return true;
}

}  // namespace Engine
}  // namespace Saiga
