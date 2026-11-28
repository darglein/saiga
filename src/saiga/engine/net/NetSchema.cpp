/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/net/NetSchema.h"

#include "saiga/engine/core/Hash.h"
#include "saiga/engine/net/WireFormat.h"
#include "saiga/engine/world/World.h"

#include <algorithm>
#include <cstring>

namespace Saiga
{
namespace Engine
{
const char* netSchemaSectionName(NetSchemaSection section)
{
    switch (section)
    {
        case NetSchemaSection::Types:
            return "types";
        case NetSchemaSection::Messages:
            return "messages";
        case NetSchemaSection::Classes:
            return "classes";
        case NetSchemaSection::Actions:
            return "actions";
        case NetSchemaSection::Templates:
            return "templates";
        case NetSchemaSection::Flags:
            return "flags";
    }
    return "?";
}

NetSchema NetSchema::fromWorld(const World& world)
{
    NetSchema s;
    const Registry& reg = world.registry();
    std::vector<SchemaType> nested;
    auto addType = [&](const SchemaType& t, uint8_t replicate, uint8_t stateKind)
    {
        Type x;
        x.kind      = t.kind;
        x.name      = t.name;
        x.id        = t.id;
        x.version   = t.version;
        x.replicate = replicate;
        x.stateKind = stateKind;
        x.fields    = t.fields;
        s.types.push_back(std::move(x));
    };
    // Every registered component in ID order (Transform first), then the events in ID order, then the nested structs
    // by name
    for (size_t id = 0; id < kMaxComponents; ++id)
    {
        const Registry::ComponentInfo& info = reg.componentInfo(ComponentId(id));
        if (!info.registered) continue;
        const uint8_t stateKind = info.global ? 1 : info.spaceState ? 2 : 0;
        if (id == Transform::id)
            addType(describeType<Transform>(SchemaKind::Component, Transform::id), uint8_t(info.replicate), stateKind);
        else
        {
            const PoolBase* p = reg.poolById(ComponentId(id));
            addType(p->describe(), uint8_t(info.replicate), stateKind);
            p->describeNested(nested);
        }
    }
    for (const auto& q : world.events.all())
        if (q)
        {
            addType(q->describe(), uint8_t(q->replicate), 0);
            q->describeNested(nested);
        }
    // Nested structs by their whole definition: two different structs of one name are both kept (a layout change of
    // either changes the hash); only identical definitions (one struct used by several components) are merged. The
    // order is total, so the encoding is deterministic.
    auto key = [](const SchemaType& t)
    {
        std::string k = t.name + '\0' + std::to_string(t.version);
        for (const FieldInfo& f : t.fields)
        {
            uint64_t q = 0;
            static_assert(sizeof(q) == sizeof(f.quantize), "the quantization step is a double");
            std::memcpy(&q, &f.quantize, sizeof(q));  // exact (to_string rounds to 6 decimals)
            k += '\0' + f.name + '\0' + f.type + '\0' + std::to_string(q);
            for (const std::string& v : f.enumValues) k += '\1' + v;
        }
        return k;
    };
    std::vector<std::pair<std::string, size_t>> order;
    for (size_t i = 0; i < nested.size(); ++i) order.push_back({key(nested[i]), i});
    std::sort(order.begin(), order.end());
    for (size_t i = 0; i < order.size(); ++i)
        if (i == 0 || order[i].first != order[i - 1].first) addType(nested[order[i].second], 0, 0);
    for (const ClassList::ClassInfo& c : world.classList().all())
        if (c.factory) s.classes.push_back({uint32_t(c.id), c.name});
    std::vector<MessageList::Entry> messages = world.messageList().entries();
    std::sort(messages.begin(), messages.end(),
              [](const MessageList::Entry& a, const MessageList::Entry& b) { return a.id < b.id; });
    for (const MessageList::Entry& e : messages)
        s.messages.push_back(Message{e.id, e.name, e.version, uint8_t(e.reliability), uint8_t(e.channel), e.raw});
    for (const ActionList::Action& a : world.actionList().actions) s.actions.push_back({a.id, a.name});
    std::sort(s.actions.begin(), s.actions.end());
    for (const ActionList::Held& h : world.actionList().held) s.held.push_back({h.bit, h.name});
    std::sort(s.held.begin(), s.held.end());
    for (const SpaceTemplate& t : world.worldConfig().spaceTemplates)
        s.templates.push_back(Template{t.name, uint8_t(t.kind), t.baseSpacing, t.sizeChunks, uint16_t(t.generator)});
    s.worldFlags       = world.streaming() ? 1u : 0u;
    s.maxChunksPerPeer = uint32_t(world.worldConfig().network.maxChunksPerPeer);
    return s;
}

namespace
{
void putDouble(std::vector<uint8_t>& out, double v)
{
    uint64_t bits = 0;
    std::memcpy(&bits, &v, sizeof(bits));
    putU64(out, bits);
}
}  // namespace

void NetSchema::encodeSection(NetSchemaSection section, std::vector<uint8_t>& out) const
{
    switch (section)
    {
        case NetSchemaSection::Types:
            writeVarint(out, types.size());
            for (const Type& t : types)
            {
                putU8(out, uint8_t(t.kind));
                putString(out, t.name);
                putU32(out, t.id);
                putU32(out, t.version);
                putU8(out, t.replicate);
                putU8(out, t.stateKind);
                writeVarint(out, t.fields.size());
                for (const FieldInfo& f : t.fields)
                {
                    putString(out, f.name);
                    putString(out, f.type);
                    writeVarint(out, f.enumValues.size());
                    for (const std::string& v : f.enumValues) putString(out, v);
                    putDouble(out, f.quantize);
                }
            }
            break;
        case NetSchemaSection::Messages:
            writeVarint(out, messages.size());
            for (const Message& m : messages)
            {
                putU16(out, m.id);
                putString(out, m.name);
                putU32(out, m.version);
                putU8(out, m.reliability);
                putU8(out, m.channel);
                putU8(out, m.raw ? 1 : 0);
            }
            break;
        case NetSchemaSection::Classes:
            writeVarint(out, classes.size());
            for (const auto& c : classes)
            {
                putU32(out, c.first);
                putString(out, c.second);
            }
            break;
        case NetSchemaSection::Actions:
            writeVarint(out, actions.size());
            for (const auto& a : actions)
            {
                putU32(out, a.first);
                putString(out, a.second);
            }
            writeVarint(out, held.size());
            for (const auto& h : held)
            {
                putU32(out, uint32_t(h.first));
                putString(out, h.second);
            }
            break;
        case NetSchemaSection::Templates:
            writeVarint(out, templates.size());
            for (const Template& t : templates)
            {
                putString(out, t.name);
                putU8(out, t.kind);
                putI32(out, t.baseSpacing);
                putI32(out, t.sizeChunks);
                putU16(out, t.generator);
            }
            break;
        case NetSchemaSection::Flags:
            putU32(out, worldFlags);
            putU32(out, maxChunksPerPeer);
            break;
    }
}

void NetSchema::encode(std::vector<uint8_t>& out) const
{
    for (size_t i = 0; i < kNetSchemaSections; ++i) encodeSection(NetSchemaSection(i), out);
}

std::array<uint64_t, kNetSchemaSections> NetSchema::sectionHashes() const
{
    std::array<uint64_t, kNetSchemaSections> h{};
    std::vector<uint8_t> bytes;
    for (size_t i = 0; i < kNetSchemaSections; ++i)
    {
        bytes.clear();
        encodeSection(NetSchemaSection(i), bytes);
        h[i] = fnv1a64Bytes(bytes.data(), bytes.size());
    }
    return h;
}

uint64_t netSchemaHashOf(const std::array<uint64_t, kNetSchemaSections>& sections)
{
    std::vector<uint8_t> bytes;
    for (uint64_t h : sections) putU64(bytes, h);
    return fnv1a64Bytes(bytes.data(), bytes.size());
}

uint64_t NetSchema::hash() const
{
    return netSchemaHashOf(sectionHashes());
}

}  // namespace Engine
}  // namespace Saiga
