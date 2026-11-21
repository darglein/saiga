/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Replication (§13.3, §13.4; S16): the host's phase-8 replication with staged bookkeeping, the client's handlers, owner
// state and owner commands, the write detector.

#include "saiga/engine/net/Replication.h"

#include "saiga/engine/components/LocalTransform.h"
#include "saiga/engine/components/NetLocal.h"
#include "saiga/engine/components/Parent.h"
#include "saiga/engine/components/Persistent.h"
#include "saiga/engine/components/Player.h"
#include "saiga/engine/components/RenderInstance.h"
#include "saiga/engine/components/RenderPose.h"
#include "saiga/engine/core/Hash.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/gameplay/Behaviour.h"
#include "saiga/engine/net/EngineMessages.h"
#include "saiga/engine/net/NetOwner.h"
#include "saiga/engine/world/World.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <zstd.h>

namespace Saiga
{
namespace Engine
{
namespace
{
constexpr uint8_t kChangeSet            = 0;
constexpr uint8_t kChangeRemove         = 1;
constexpr uint8_t kChangePresenceAdd    = 2;
constexpr uint8_t kChangePresenceRemove = 3;
constexpr uint8_t kSnapshotCompressed   = 1;
// Worst case of the snapshot header: flags, epoch, host step, sequence, part, parts
constexpr size_t kMaxSnapshotHeader = 1 + 5 + 10 + 10 + 5 + 5;
constexpr size_t kMaxSnapshotBody   = kMaxMessagePayload - kMaxSnapshotHeader;
constexpr size_t kMaxOwnerHeader    = 10 + 10 + 5;
constexpr size_t kMaxDecompressed   = size_t(1) << 16;
static_assert(Replication::kMaxEntitySnapshotBytes <= kMaxSnapshotBody, "an entity record must fit a message");

uint64_t componentKey(uint32_t index, ComponentId id)
{
    return (uint64_t(index) << 8) | id;
}

bool replicatedMode(Replicate r)
{
    return r == Replicate::Host || r == Replicate::HostReliable || r == Replicate::OwnerAuthoritative;
}

// A component a client default-constructs from the spawn's presence list: peer-local (Never), saved, not engine
// bookkeeping
bool presenceComponent(const Registry& reg, ComponentId id)
{
    const Registry::ComponentInfo& info = reg.componentInfo(id);
    return info.registered && info.replicate == Replicate::Never && info.saved == Saved::Yes && id != Behaviour::id &&
           id != NetLocal::id && id != Transform::id;
}

void putU8(std::vector<uint8_t>& out, uint8_t v)
{
    out.push_back(v);
}
void putU16(std::vector<uint8_t>& out, uint16_t v)
{
    out.push_back(uint8_t(v & 0xFF));
    out.push_back(uint8_t(v >> 8));
}

// Reads untrusted input; every error is remembered (never asserts)
struct ByteReader
{
    const uint8_t* d;
    size_t n;
    size_t pos = 0;
    bool ok    = true;
    ByteReader(const uint8_t* data, size_t size) : d(data), n(size) {}
    uint64_t varint()
    {
        uint64_t v = 0;
        if (ok && !readVarint(d, n, pos, v)) ok = false;
        return ok ? v : 0;
    }
    uint8_t u8()
    {
        if (!ok || pos + 1 > n)
        {
            ok = false;
            return 0;
        }
        return d[pos++];
    }
    uint16_t u16()
    {
        const uint16_t lo = u8();
        const uint16_t hi = u8();
        return uint16_t(lo | (hi << 8));
    }
    const uint8_t* bytes(uint64_t len)
    {
        if (!ok || len > n - pos)
        {
            ok = false;
            return nullptr;
        }
        const uint8_t* p = d + pos;
        pos += size_t(len);
        return p;
    }
    bool atEnd() const { return ok && pos == n; }
};

struct ComponentBytes
{
    ComponentId id;
    const uint8_t* data;
    size_t size;
};

// [count] {[component u8][length][bytes]}
bool readComponents(ByteReader& r, std::vector<ComponentBytes>& out)
{
    const uint64_t count = r.varint();
    if (!r.ok || count > 256) return false;
    out.clear();
    for (uint64_t i = 0; i < count && r.ok; ++i)
    {
        const ComponentId id = r.u8();
        const uint64_t len   = r.varint();
        const uint8_t* p     = r.bytes(len);
        if (r.ok) out.push_back(ComponentBytes{id, p, size_t(len)});
    }
    return r.ok;
}

// The parsed entity record of a snapshot or owner state: [NetId][counter u8][components]
struct EntityRecord
{
    NetId netId     = 0;
    uint8_t counter = 0;
    std::vector<ComponentBytes> components;
};
bool readRecord(ByteReader& r, EntityRecord& out)
{
    const uint64_t id = r.varint();
    out.counter       = r.u8();
    if (!r.ok || id == 0 || id > 0xFFFFFFFFull) return false;
    out.netId = NetId(id);
    return readComponents(r, out.components);
}

// Entering an engine handler: structural changes apply at once (as during a flush), so the spawn is one unit
struct Immediate
{
    bool& flag;
    bool old;
    explicit Immediate(bool& f) : flag(f), old(f) { flag = true; }
    ~Immediate() { flag = old; }
};
}  // namespace

Replication::Replication(World& world) : w(world)
{
    if (w.isHost()) logConsumer = w.log.addConsumer();
}

Replication::~Replication()
{
    if (zstd) ZSTD_freeCCtx(static_cast<ZSTD_CCtx*>(zstd));
}

// ---- NetIds -------------------------------------------------------------------------------------------------------

const Replication::HostSlot* Replication::hostSlot(Entity e) const
{
    if (e.index == 0 || e.index >= host.size()) return nullptr;
    const HostSlot& s = host[e.index];
    return s.generation == e.generation ? &s : nullptr;
}

Replication::HostSlot* Replication::hostSlot(Entity e)
{
    return const_cast<HostSlot*>(static_cast<const Replication*>(this)->hostSlot(e));
}

NetId Replication::netIdOf(Entity e) const
{
    if (w.isHost())
    {
        const HostSlot* s = hostSlot(e);
        return s && s->spawned ? s->netId : 0;
    }
    if (e.index < clientNetIdOf.size() && clientNetIdOf[e.index].first == e.generation && e.index != 0)
        return clientNetIdOf[e.index].second;
    return 0;
}

Entity Replication::entityOfNetId(NetId id) const
{
    if (id == 0) return Entity{};
    if (w.isHost())
    {
        auto it = hostByNetId.find(id);
        return it == hostByNetId.end() || !w.reg.alive(it->second) ? Entity{} : it->second;
    }
    auto it = clientByNetId.find(id);
    return it == clientByNetId.end() || !w.reg.alive(it->second.entity) ? Entity{} : it->second.entity;
}

bool Replication::spawned(Entity e) const
{
    if (!w.isHost()) return netIdOf(e) != 0;
    const HostSlot* s = hostSlot(e);
    return s && s->spawned;
}

uint8_t Replication::commandCounter(Entity root) const
{
    if (w.isHost())
    {
        const HostSlot* s = hostSlot(root);
        return s ? s->command : 0;
    }
    auto it = clientByNetId.find(netIdOf(root));
    return it == clientByNetId.end() ? 0 : it->second.command;
}

uint8_t Replication::snapCounter(Entity root) const
{
    if (w.isHost())
    {
        const HostSlot* s = hostSlot(root);
        return s ? s->snap : 0;
    }
    auto it = clientByNetId.find(netIdOf(root));
    return it == clientByNetId.end() ? 0 : it->second.snap;
}

std::vector<NetId> Replication::netIds() const
{
    std::vector<NetId> ids;
    if (w.isHost())
    {
        for (const auto& kv : hostByNetId)
            if (w.reg.alive(kv.second)) ids.push_back(kv.first);
    }
    else
        for (const auto& kv : clientByNetId)
            if (w.reg.alive(kv.second.entity)) ids.push_back(kv.first);
    std::sort(ids.begin(), ids.end());
    return ids;
}

InterpolationBuffer* Replication::buffer(Entity e)
{
    auto it = buffers.find(e.index);
    return it == buffers.end() || it->second.first != e.generation ? nullptr : &it->second.second;
}

void Replication::mapClient(NetId id, Entity e, uint8_t snap, uint8_t command)
{
    clientByNetId[id] = ClientEntry{e, snap, command};
    if (clientNetIdOf.size() <= e.index) clientNetIdOf.resize(size_t(e.index) + 1, {0, 0});
    clientNetIdOf[e.index] = {e.generation, id};
}

void Replication::unmapClient(Entity e)
{
    const NetId id = netIdOf(e);
    if (id == 0) return;
    clientByNetId.erase(id);
    clientNetIdOf[e.index] = {0, 0};
    pendingParents.erase(id);
    buffers.erase(e.index);
    for (size_t c = 0; c < kMaxComponents; ++c) written.erase(componentKey(e.index, ComponentId(c)));
}

Replication::ClientEntry* Replication::clientEntry(Entity e)
{
    auto it = clientByNetId.find(netIdOf(e));
    return it == clientByNetId.end() ? nullptr : &it->second;
}

// ---- staging (host) -----------------------------------------------------------------------------------------------

Replication::HostSlot& Replication::writeSlot(uint32_t index)
{
    if (host.size() <= index) host.resize(std::max<size_t>(size_t(index) + 1, size_t(w.reg.slotCount())));
    if (w.inStep()) slotUndo.push_back(SlotUndo{w.phase, index, host[index]});
    return host[index];
}

void Replication::setHash(uint64_t key, uint64_t value)
{
    auto it = reliableHash.find(key);
    if (w.inStep())
        hashUndo.push_back(HashUndo{w.phase, key, it != reliableHash.end(), it == reliableHash.end() ? 0 : it->second});
    reliableHash[key] = value;
}

void Replication::eraseHash(uint64_t key)
{
    auto it = reliableHash.find(key);
    if (it == reliableHash.end()) return;
    if (w.inStep()) hashUndo.push_back(HashUndo{w.phase, key, true, it->second});
    reliableHash.erase(it);
}

void Replication::setNetIdEntry(NetId id, Entity e)
{
    auto it = hostByNetId.find(id);
    if (w.inStep())
        netIdUndo.push_back(
            NetIdUndo{w.phase, id, it != hostByNetId.end(), it == hostByNetId.end() ? Entity{} : it->second});
    if (e.valid())
        hostByNetId[id] = e;
    else if (it != hostByNetId.end())
        hostByNetId.erase(it);
}

void Replication::commit()
{
    slotUndo.clear();
    hashUndo.clear();
    netIdUndo.clear();
    teleportUndo.clear();
    replicatedThisStep = false;
}

void Replication::abort(Phase aborted)
{
    // In reverse order, only what the aborted phase staged (finished phases keep their changes, as their messages)
    for (size_t i = slotUndo.size(); i-- > 0;)
        if (slotUndo[i].phase == aborted) host[slotUndo[i].index] = slotUndo[i].old;
    for (size_t i = hashUndo.size(); i-- > 0;)
        if (hashUndo[i].phase == aborted)
        {
            if (hashUndo[i].existed)
                reliableHash[hashUndo[i].key] = hashUndo[i].old;
            else
                reliableHash.erase(hashUndo[i].key);
        }
    for (size_t i = netIdUndo.size(); i-- > 0;)
        if (netIdUndo[i].phase == aborted)
        {
            if (netIdUndo[i].existed)
                hostByNetId[netIdUndo[i].id] = netIdUndo[i].old;
            else
                hostByNetId.erase(netIdUndo[i].id);
        }
    for (size_t i = teleportUndo.size(); i-- > 0;)
        if (teleportUndo[i].phase == aborted)
        {
            TeleportUndo& u = teleportUndo[i];
            if (u.hadRecord)
                ownerRecords[u.index] = u.record;
            else
                ownerRecords.erase(u.index);
            if (u.hadBuffer)
                buffers[u.index] = u.buffer;
            else
                buffers.erase(u.index);
        }
    teleportUndo.erase(std::remove_if(teleportUndo.begin(), teleportUndo.end(),
                                      [&](const TeleportUndo& u) { return u.phase == aborted; }),
                       teleportUndo.end());
    for (TeleportUndo& u : teleportUndo) u.phase = Phase::None;
    slotUndo.erase(
        std::remove_if(slotUndo.begin(), slotUndo.end(), [&](const SlotUndo& u) { return u.phase == aborted; }),
        slotUndo.end());
    hashUndo.erase(
        std::remove_if(hashUndo.begin(), hashUndo.end(), [&](const HashUndo& u) { return u.phase == aborted; }),
        hashUndo.end());
    netIdUndo.erase(
        std::remove_if(netIdUndo.begin(), netIdUndo.end(), [&](const NetIdUndo& u) { return u.phase == aborted; }),
        netIdUndo.end());
    // The kept entries belong to no phase any more (a later abort of their phase must not undo them)
    for (SlotUndo& u : slotUndo) u.phase = Phase::None;
    for (HashUndo& u : hashUndo) u.phase = Phase::None;
    for (NetIdUndo& u : netIdUndo) u.phase = Phase::None;
    if (aborted == Phase::Extract && replicatedThisStep)
    {
        // The records the replication consumed are read again in the next step (the log is truncated only after a
        // completed step)
        w.log.rewind(logConsumer, logCursorBefore);
        nextNetId   = nextNetIdBefore;
        snapshotSeq = snapshotSeqBefore;
    }
    replicatedThisStep = false;
}

void Replication::onHostLoad()
{
    if (!w.isHost()) return;
    // A new epoch: every loaded entity is spawned again (with new NetIds; NetIds are never reused), clients drop the
    // old state and the snapshots in flight
    currentEpoch++;
    host.clear();
    hostByNetId.clear();
    reliableHash.clear();
    ownerRecords.clear();
    buffers.clear();
    written.clear();
    slotUndo.clear();
    hashUndo.clear();
    netIdUndo.clear();
    if (w.net)
        for (uint8_t p : w.connectedPeers()) w.send(p, ResyncMessage{currentEpoch});
}

// ---- encoding -----------------------------------------------------------------------------------------------------

void Replication::encodeComponent(Entity e, ComponentId id, std::vector<uint8_t>& out) const
{
    NetWriter writer(out, this);
    if (id == Transform::id)
    {
        Transform t = w.reg.read<Transform>(e);
        t.visit(writer);
    }
    else
        w.reg.poolById(id)->netAppend(e.index, writer);
    if (!writer.ok())
        SAIGA_EXIT_ERROR("replication: " + std::string(w.reg.componentInfo(id).name) + " of " + describe(e) + ": " +
                         writer.error());
}

std::vector<NetLeaf> Replication::flattenComponent(Entity e, ComponentId id) const
{
    std::vector<NetLeaf> out;
    if (id == Transform::id)
    {
        Transform t = w.reg.read<Transform>(e);
        NetFlattener f(out, this);
        t.visit(f);
    }
    else
        w.reg.poolById(id)->netFlatten(e.index, out, this);
    return out;
}

std::string Replication::describe(Entity e) const
{
    std::string s  = "entity " + std::to_string(e.index) + ":" + std::to_string(e.generation);
    const NetId id = netIdOf(e);
    if (id) s += " (NetId " + std::to_string(id) + ")";
    return s;
}

bool Replication::hostReplicated(Entity e) const
{
    // Not NetLocal, itself or an ancestor; the chain is bounded (a broken registry cannot loop)
    const Registry& reg = w.reg;
    Entity a            = e;
    for (uint32_t depth = 0; depth <= reg.slotCount(); ++depth)
    {
        if (!reg.alive(a)) return false;
        if (reg.registered(NetLocal::id) && reg.has<NetLocal>(a)) return false;
        const Parent* p = reg.tryRead<Parent>(a);
        if (!p) return true;
        a = p->entity;
    }
    return false;
}

int Replication::depthOf(Entity e) const
{
    int depth = 0;
    Entity a  = e;
    while (const Parent* p = w.reg.tryRead<Parent>(a))
    {
        if (++depth > int(w.reg.slotCount())) break;
        a = p->entity;
    }
    return depth;
}

void Replication::encodeSpawn(Entity e, std::vector<uint8_t>& out)
{
    const Registry& reg = w.reg;
    const HostSlot& s   = host[e.index];
    out.clear();
    writeVarint(out, s.netId);
    const Behaviour* b = reg.registered(Behaviour::id) ? reg.tryRead<Behaviour>(e) : nullptr;
    putU16(out, b ? b->classId : ClassId(0));
    putU8(out, s.owner);
    const Parent* parent = reg.tryRead<Parent>(e);
    writeVarint(out, parent ? netIdOf(parent->entity) : 0);
    putU8(out, s.snap);
    putU8(out, s.command);
    // The LocalTransform with the parent (the hierarchy change is atomic)
    scratch.clear();
    if (parent)
    {
        NetWriter lw(scratch, this);
        LocalTransform identity;
        const LocalTransform* local = reg.tryRead<LocalTransform>(e);
        lw.write(local ? *local : identity);
    }
    writeVarint(out, scratch.size());
    out.insert(out.end(), scratch.begin(), scratch.end());
    // Every replicated component (the child's Transform is derived), then the presence list of the peer-local ones
    const ComponentMask& m = reg.entityMask(e);
    const bool state       = (reg.flags(e) & (Registry::kWorldState | Registry::kSpaceState)) != 0;
    size_t count = 0, presence = 0;
    for (size_t id = 0; id < kMaxComponents; ++id)
    {
        if (!m.test(ComponentId(id))) continue;
        const Replicate mode = reg.componentInfo(ComponentId(id)).replicate;
        if (replicatedMode(mode) && id != Parent::id && !(id == Transform::id && (parent || state))) count++;
        if (presenceComponent(reg, ComponentId(id))) presence++;
    }
    writeVarint(out, count);
    for (size_t id = 0; id < kMaxComponents; ++id)
    {
        if (!m.test(ComponentId(id))) continue;
        const Replicate mode = reg.componentInfo(ComponentId(id)).replicate;
        if (!replicatedMode(mode) || id == Parent::id || (id == Transform::id && (parent || state))) continue;
        scratch.clear();
        encodeComponent(e, ComponentId(id), scratch);
        putU8(out, uint8_t(id));
        writeVarint(out, scratch.size());
        out.insert(out.end(), scratch.begin(), scratch.end());
        if (mode == Replicate::HostReliable)
            setHash(componentKey(e.index, ComponentId(id)), fnv1a64Bytes(scratch.data(), scratch.size()));
    }
    writeVarint(out, presence);
    for (size_t id = 0; id < kMaxComponents; ++id)
        if (m.test(ComponentId(id)) && presenceComponent(reg, ComponentId(id))) putU8(out, uint8_t(id));
    counts.spawnsSent++;
}

void Replication::encodeRecord(Entity e, const HostSlot& s, bool withOwnerAuthoritative, std::vector<uint8_t>& out)
{
    const Registry& reg    = w.reg;
    const ComponentMask& m = reg.entityMask(e);
    const bool child       = m.test(Parent::id);
    const bool state       = (reg.flags(e) & (Registry::kWorldState | Registry::kSpaceState)) != 0;
    out.clear();
    writeVarint(out, s.netId);
    // The snap counter of the root (children follow their root's teleports in phase 6)
    Entity root = e;
    for (int depth = 0; depth <= int(reg.slotCount()); ++depth)
    {
        const Parent* p = reg.tryRead<Parent>(root);
        if (!p) break;
        root = p->entity;
    }
    const HostSlot* rs = hostSlot(root);
    putU8(out, rs ? rs->snap : 0);
    const size_t countAt = out.size();
    putU8(out, 0);  // the count (< 128 components fit one varint byte; patched below)
    size_t count = 0, largest = 0;
    ComponentId largestId = 0;
    for (size_t id = 0; id < kMaxComponents; ++id)
    {
        if (!m.test(ComponentId(id))) continue;
        const Replicate mode = reg.componentInfo(ComponentId(id)).replicate;
        if (mode != Replicate::Host && mode != Replicate::OwnerAuthoritative) continue;
        if (mode == Replicate::OwnerAuthoritative && !withOwnerAuthoritative) continue;
        if (id == Parent::id || (id == Transform::id && (child || state))) continue;
        scratch.clear();
        encodeComponent(e, ComponentId(id), scratch);
        putU8(out, uint8_t(id));
        writeVarint(out, scratch.size());
        out.insert(out.end(), scratch.begin(), scratch.end());
        if (scratch.size() > largest)
        {
            largest   = scratch.size();
            largestId = ComponentId(id);
        }
        count++;
    }
    SAIGA_ASSERT(count < 128, "replication: more than 127 snapshot components on one entity");
    out[countAt] = uint8_t(count);
    // §13.4: the unreliable state of one entity is bounded (a larger component must be HostReliable); every build
    if (out.size() > kMaxEntitySnapshotBytes)
        SAIGA_EXIT_ERROR("replication: the snapshot state of " + describe(e) + " has " + std::to_string(out.size()) +
                         " bytes (at most " + std::to_string(kMaxEntitySnapshotBytes) + "); its largest component is " +
                         reg.componentInfo(largestId).name + " (" + std::to_string(largest) +
                         " bytes): register it HostReliable");
}

void Replication::sendToAll(const std::vector<uint8_t>& peers, MessageId id, const std::vector<uint8_t>& payload)
{
    for (uint8_t p : peers) w.sendRaw(p, id, payload.data(), payload.size());
}

// ---- host replication (phase 8) -----------------------------------------------------------------------------------

void Replication::hostReplicate(bool networkOnly)
{
    (void)networkOnly;
    Registry& reg      = w.reg;
    StructuralLog& log = w.log;
    replicatedThisStep = true;
    logCursorBefore    = log.cursor(logConsumer);
    nextNetIdBefore    = nextNetId;
    snapshotSeqBefore  = snapshotSeq;

    // A peer that joined in this receive gets the full spawn set of the current state; the others the regular messages
    regularPeers.clear();
    joinedPeers.clear();
    for (uint8_t p : w.connectedPeers())
    {
        const auto& joined = w.peersJoined();
        if (std::find(joined.begin(), joined.end(), p) != joined.end())
            joinedPeers.push_back(p);
        else
            regularPeers.push_back(p);
    }
    std::sort(regularPeers.begin(), regularPeers.end());
    std::sort(joinedPeers.begin(), joinedPeers.end());
    const bool anyPeer = !regularPeers.empty() || !joinedPeers.empty();

    // 1. The structural log: creations (candidates), destroys, hierarchy and component changes of spawned entities
    candidates.clear();
    hierarchyNow.clear();
    changedNow.clear();
    std::vector<NetId> destroyed;
    log.consume(
        logConsumer,
        [&](const StructuralLog::Record& r)
        {
            const Entity e = r.entity;
            if (!r.added) eraseHash(componentKey(e.index, r.component));
            if (r.component == Transform::id)
            {
                if (r.added)
                {
                    candidates.push_back(e);
                    return;
                }
                HostSlot* s = hostSlot(e);
                if (s && s->spawned)
                {
                    // Destroyed after its spawn (an entity spawned and destroyed in one step sends nothing)
                    destroyed.push_back(s->netId);
                    const NetId id = s->netId;
                    HostSlot& ws   = writeSlot(e.index);
                    ws.spawned     = false;
                    ws.netId       = 0;
                    setNetIdEntry(id, Entity{});
                }
                return;
            }
            HostSlot* s = hostSlot(e);
            if (!s || !s->spawned)
            {
                // A local entity may become replicated (reparented out of a NetLocal subtree, NetLocal removed)
                if ((r.component == Parent::id || r.component == NetLocal::id) && reg.alive(e)) candidates.push_back(e);
                return;
            }
            if (r.component == NetLocal::id)
            {
                SAIGA_ASSERT(false, "NetLocal added to or removed from " + describe(e) + " after its spawn was sent");
                return;
            }
            if (r.component == Parent::id)
                hierarchyNow.push_back(e);
            else if (replicatedMode(reg.componentInfo(r.component).replicate) || presenceComponent(reg, r.component))
                changedNow.push_back({e, r.component});
        });

    // 2. New replicated entities: roots first, then descendants by depth (a parent is always spawned first)
    std::vector<std::pair<int, Entity>> fresh;
    {
        std::sort(candidates.begin(), candidates.end());
        candidates.erase(std::unique(candidates.begin(), candidates.end()), candidates.end());
        std::vector<Entity> queue;
        for (Entity e : candidates)
        {
            const HostSlot* s = hostSlot(e);
            if (!reg.alive(e) || (s && s->spawned) || !hostReplicated(e)) continue;
            queue.push_back(e);
        }
        // The not yet replicated descendants of a new entity become replicated with it
        for (size_t i = 0; i < queue.size(); ++i)
        {
            if (w.childCount(queue[i]) == 0) continue;
            for (Entity x : w.collectDoomed(queue[i]))
            {
                const HostSlot* s = hostSlot(x);
                if (x == queue[i] || (s && s->spawned) || !hostReplicated(x)) continue;
                if (std::find(queue.begin(), queue.end(), x) == queue.end()) queue.push_back(x);
            }
        }
        for (Entity e : queue) fresh.push_back({depthOf(e), e});
        std::sort(fresh.begin(), fresh.end(), [](const std::pair<int, Entity>& a, const std::pair<int, Entity>& b)
                  { return a.first != b.first ? a.first < b.first : a.second.index < b.second.index; });
    }
    for (const auto& f : fresh)
    {
        const Entity e = f.second;
        HostSlot& s    = writeSlot(e.index);
        s              = HostSlot{};
        s.generation   = e.generation;
        s.spawned      = true;
        s.owner        = w.ownerOf(e);
        if (reg.flags(e) & Registry::kWorldState)
            s.netId = kWorldStateNetId;
        else if (reg.flags(e) & Registry::kSpaceState)
        {
            const auto it = std::find(w.spaceStates.begin(), w.spaceStates.end(), e);
            s.netId       = kFirstSpaceStateNetId + NetId(it - w.spaceStates.begin());
        }
        else
            s.netId = nextNetId++;
        setNetIdEntry(s.netId, e);
        if (!regularPeers.empty())
        {
            encodeSpawn(e, record);
            sendToAll(regularPeers, MessageId(EngineMessage::Spawn), record);
        }
    }
    auto isFresh = [&](Entity e)
    {
        for (const auto& f : fresh)
            if (f.second == e) return true;
        return false;
    };

    // 3. Hierarchy changes (reparenting)
    std::sort(hierarchyNow.begin(), hierarchyNow.end());
    hierarchyNow.erase(std::unique(hierarchyNow.begin(), hierarchyNow.end()), hierarchyNow.end());
    for (Entity e : hierarchyNow)
    {
        const HostSlot* s = hostSlot(e);
        if (!reg.alive(e) || !s || !s->spawned || isFresh(e)) continue;
        const Parent* p = reg.tryRead<Parent>(e);
        HierarchyMessage m;
        m.netId  = s->netId;
        m.parent = p ? netIdOf(p->entity) : 0;
        if (p && m.parent == 0)
        {
            SAIGA_ASSERT(false, describe(e) + " was reparented under a local entity (NetLocal subtree)");
            continue;
        }
        // Reparenting never changes the owner (§13.3)
        if (w.ownerOf(e) != s->owner)
        {
            // Refused by add/remove<Parent> (World::netChangeRefused); a hierarchy that changed the owner is not sent
            SAIGA_ASSERT(false, describe(e) + ": reparenting changed its owner from peer " + std::to_string(s->owner) +
                                    " to " + std::to_string(w.ownerOf(e)));
            continue;
        }
        if (const LocalTransform* l = reg.tryRead<LocalTransform>(e)) m.local = *l;
        for (uint8_t peer : regularPeers)
        {
            w.send(peer, m);
            counts.hierarchySent++;
        }
    }

    // 4. Destroys (after the hierarchy changes: a child moved out of a destroyed parent leaves it before the client's
    // cascade)
    for (NetId id : destroyed)
        for (uint8_t p : regularPeers)
        {
            w.send(p, DestroyMessage{id});
            counts.destroysSent++;
        }

    // 5. Component changes: additions and removals, and HostReliable components whose bytes changed
    std::sort(changedNow.begin(), changedNow.end());
    changedNow.erase(std::unique(changedNow.begin(), changedNow.end()), changedNow.end());
    auto sendChange = [&](Entity e, NetId id, uint8_t kind, ComponentId c)
    {
        record.clear();
        writeVarint(record, id);
        putU8(record, kind);
        putU8(record, c);
        if (kind == kChangeSet)
        {
            scratch.clear();
            encodeComponent(e, c, scratch);
            record.insert(record.end(), scratch.begin(), scratch.end());
            if (reg.componentInfo(c).replicate == Replicate::HostReliable)
                setHash(componentKey(e.index, c), fnv1a64Bytes(scratch.data(), scratch.size()));
        }
        sendToAll(regularPeers, MessageId(EngineMessage::ComponentChange), record);
        counts.componentChangesSent += regularPeers.size();
    };
    if (!regularPeers.empty())
    {
        for (const auto& c : changedNow)
        {
            const Entity e    = c.first;
            const HostSlot* s = hostSlot(e);
            if (!reg.alive(e) || !s || !s->spawned || isFresh(e)) continue;
            const bool has      = reg.entityMask(e).test(c.second);
            const bool presence = presenceComponent(reg, c.second);
            sendChange(
                e, s->netId,
                presence ? (has ? kChangePresenceAdd : kChangePresenceRemove) : (has ? kChangeSet : kChangeRemove),
                c.second);
        }
        for (size_t id = 1; id < kMaxComponents; ++id)
        {
            const Registry::ComponentInfo& info = reg.componentInfo(ComponentId(id));
            if (!info.registered || info.replicate != Replicate::HostReliable) continue;
            PoolBase* pool = reg.poolById(ComponentId(id));
            for (size_t k = 0; k < pool->size(); ++k)
            {
                const Entity e    = pool->denseEntities()[k];
                const HostSlot* s = hostSlot(e);
                if (!s || !s->spawned || !reg.alive(e)) continue;
                scratch.clear();
                encodeComponent(e, ComponentId(id), scratch);
                const uint64_t h   = fnv1a64Bytes(scratch.data(), scratch.size());
                const uint64_t key = componentKey(e.index, ComponentId(id));
                auto it            = reliableHash.find(key);
                if (it != reliableHash.end() && it->second == h) continue;
                sendChange(e, s->netId, kChangeSet, ComponentId(id));
            }
        }
    }

    if (!anyPeer) return;

    // 6. Snap counters: a teleport of a spawned root in this step (counted once, also over stepNetworkOnly)
    for (uint32_t i = 1; i < host.size(); ++i)
    {
        const HostSlot& s = host[i];
        if (!s.spawned) continue;
        const Entity e{i, s.generation};
        if (!reg.alive(e) || reg.entityMask(e).test(Parent::id)) continue;
        if (reg.read<Transform>(e).wasTeleported() && s.snapCountedAt != w.steps)
        {
            HostSlot& ws = writeSlot(i);
            ws.snap++;
            ws.snapCountedAt = w.steps;
        }
    }

    // 7. Joined peers: the epoch of the host's last load (the joiner starts at 0 and would drop the snapshots), then
    // the full spawn set (roots first, then by depth)
    if (!joinedPeers.empty())
    {
        if (currentEpoch != 0)
            for (uint8_t p : joinedPeers) w.send(p, ResyncMessage{currentEpoch});
        std::vector<std::pair<int, Entity>> all;
        for (uint32_t i = 1; i < host.size(); ++i)
        {
            const Entity e{i, host[i].generation};
            if (host[i].spawned && reg.alive(e)) all.push_back({depthOf(e), e});
        }
        std::sort(all.begin(), all.end(), [](const std::pair<int, Entity>& a, const std::pair<int, Entity>& b)
                  { return a.first != b.first ? a.first < b.first : a.second.index < b.second.index; });
        for (const auto& x : all)
        {
            encodeSpawn(x.second, record);
            sendToAll(joinedPeers, MessageId(EngineMessage::Spawn), record);
        }
    }

    // 8. One snapshot of all replicated entities to every connected peer
    std::vector<uint8_t> peers = regularPeers;
    peers.insert(peers.end(), joinedPeers.begin(), joinedPeers.end());
    std::sort(peers.begin(), peers.end());
    sendSnapshot(peers);
}

void Replication::sendSnapshot(const std::vector<uint8_t>& peers)
{
    Registry& reg = w.reg;
    // Every entity is serialized once: the full record, and for client-owned entities the owner's (without its
    // owner-authoritative components, which the owner does not get back)
    struct Item
    {
        uint8_t owner;
        size_t full, fullSize, own, ownSize;
    };
    std::vector<Item> items;
    std::vector<uint8_t> arena;
    for (uint32_t i = 1; i < host.size(); ++i)
    {
        const HostSlot& s = host[i];
        const Entity e{i, s.generation};
        if (!s.spawned || !reg.alive(e)) continue;
        Item it{s.owner, arena.size(), 0, 0, 0};
        encodeRecord(e, s, true, record);
        arena.insert(arena.end(), record.begin(), record.end());
        it.fullSize = record.size();
        it.own      = arena.size();
        if (s.owner != 0)
        {
            encodeRecord(e, s, false, record);
            arena.insert(arena.end(), record.begin(), record.end());
            it.ownSize = record.size();
        }
        items.push_back(it);
    }
    // Nothing replicated (no entity at all): no snapshot
    if (items.empty()) return;
    snapshotSeq++;
    counts.snapshotsSent++;
    if (!zstd) zstd = ZSTD_createCCtx();
    std::vector<std::pair<size_t, size_t>> bodies;  // per peer: [begin, end) of items per message
    std::vector<uint8_t> body, compressed, payload;
    for (uint8_t peer : peers)
    {
        // Split at entity boundaries
        bodies.clear();
        size_t begin = 0, bytes = 0;
        for (size_t k = 0; k < items.size(); ++k)
        {
            const size_t n = items[k].owner == peer ? items[k].ownSize : items[k].fullSize;
            if (k > begin && bytes + n > kMaxSnapshotBody)
            {
                bodies.push_back({begin, k});
                begin = k;
                bytes = 0;
            }
            bytes += n;
        }
        bodies.push_back({begin, items.size()});
        for (size_t part = 0; part < bodies.size(); ++part)
        {
            body.clear();
            for (size_t k = bodies[part].first; k < bodies[part].second; ++k)
            {
                const Item& it = items[k];
                if (it.owner == peer)
                    body.insert(body.end(), arena.begin() + std::ptrdiff_t(it.own),
                                arena.begin() + std::ptrdiff_t(it.own + it.ownSize));
                else
                    body.insert(body.end(), arena.begin() + std::ptrdiff_t(it.full),
                                arena.begin() + std::ptrdiff_t(it.full + it.fullSize));
            }
            // zstd per message (decodable alone), raw when it does not shrink
            compressed.resize(ZSTD_compressBound(body.size()));
            const size_t c = ZSTD_compressCCtx(static_cast<ZSTD_CCtx*>(zstd), compressed.data(), compressed.size(),
                                               body.data(), body.size(), 1);
            const bool useCompressed = !ZSTD_isError(c) && c < body.size();
            payload.clear();
            putU8(payload, useCompressed ? kSnapshotCompressed : 0);
            writeVarint(payload, currentEpoch);
            writeVarint(payload, w.steps);
            writeVarint(payload, snapshotSeq);
            writeVarint(payload, part);
            writeVarint(payload, bodies.size());
            if (useCompressed)
            {
                payload.insert(payload.end(), compressed.begin(), compressed.begin() + std::ptrdiff_t(c));
                counts.compressedMessages++;
            }
            else
                payload.insert(payload.end(), body.begin(), body.end());
            w.sendRaw(peer, MessageId(EngineMessage::Snapshot), payload.data(), payload.size());
            counts.snapshotMessagesSent++;
            counts.snapshotBytesSent += payload.size();
        }
    }
}

// ---- owner commands, owner state, owner poses ---------------------------------------------------------------------

void Replication::ownerTeleport(Entity root, const Vec3& position, const Quat& rotation)
{
    Transform& t        = w.reg.get<Transform>(root);
    const HostSlot* s   = hostSlot(root);
    const uint8_t owner = w.ownerOf(root);
    t.teleport(position, rotation);
    // Before the spawn or host-owned: a plain teleport (the snapshots convey it)
    if (!s || !s->spawned || owner == 0 || !w.net) return;
    HostSlot& ws = writeSlot(root.index);
    ws.command++;
    OwnerCommandMessage m;
    m.netId    = ws.netId;
    m.kind     = OwnerCommandMessage::kTeleport;
    m.position = position;
    m.rotation = rotation;
    m.counter  = ws.command;
    w.send(owner, m);
    // The owner's states in flight carry the old counter and are dropped; the stored one is stale
    auto b = buffers.find(root.index);
    if (w.inStep())
    {
        auto r = ownerRecords.find(root.index);
        TeleportUndo u{w.phase, root.index, r != ownerRecords.end(), {}, b != buffers.end(), {}};
        if (u.hadRecord) u.record = r->second;
        if (u.hadBuffer) u.buffer = b->second;
        teleportUndo.push_back(std::move(u));
    }
    ownerRecords.erase(root.index);
    if (b != buffers.end() && b->second.first == root.generation)
        b->second.second.push(double(w.steps), position, rotation, ws.command);
    noteWritten(root, Transform::id);
}

void Replication::sendPlayerAction(const PlayerAction& action)
{
    const NetId id = netIdOf(action.player);
    if (id == 0)
    {
        SAIGA_ASSERT(false, "queueAction: the player is not replicated");
        return;
    }
    PlayerActionMessage m;
    m.player = id;
    m.action = action.action;
    m.aim    = action.aim;
    w.send(0, m);
}

void Replication::ownerStateSend(bool networkOnly)
{
    if (!w.net) return;
    // The round trip for the clock, once snapshots arrive
    if (!networkOnly && hostClock.valid() && w.steps % uint64_t(HostClock::kPingInterval) == 0)
        w.send(0, PingMessage{w.steps});
    // The owned roots and their descendants with a NetId: their owner-authoritative components (not the derived
    // Transform of a child)
    Registry& reg = w.reg;
    std::vector<std::vector<uint8_t>> records;
    w.ownedBy<const Transform>(w.selfPeer)
        .each(
            [&](Entity e, const Transform&)
            {
                const NetId id = netIdOf(e);
                if (id == 0) return;
                const ComponentMask& m = reg.entityMask(e);
                const bool child       = m.test(Parent::id);
                Entity root            = e;
                while (const Parent* p = reg.tryRead<Parent>(root)) root = p->entity;
                record.clear();
                writeVarint(record, id);
                putU8(record, commandCounter(root));
                const size_t countAt = record.size();
                putU8(record, 0);
                uint8_t count = 0;
                for (size_t c = 0; c < kMaxComponents; ++c)
                {
                    if (!m.test(ComponentId(c)) ||
                        reg.componentInfo(ComponentId(c)).replicate != Replicate::OwnerAuthoritative)
                        continue;
                    if (c == Transform::id && child) continue;
                    scratch.clear();
                    encodeComponent(e, ComponentId(c), scratch);
                    putU8(record, uint8_t(c));
                    writeVarint(record, scratch.size());
                    record.insert(record.end(), scratch.begin(), scratch.end());
                    count++;
                }
                record[countAt] = count;
                if (record.size() > kMaxEntitySnapshotBytes)
                    SAIGA_EXIT_ERROR("replication: the owner state of " + describe(e) + " has " +
                                     std::to_string(record.size()) + " bytes (at most " +
                                     std::to_string(kMaxEntitySnapshotBytes) + ")");
                records.push_back(record);
            });
    if (records.empty()) return;
    const uint64_t seq   = ++ownerSeq[w.selfPeer];
    const double est     = hostClock.estimate(w.steps);
    const uint64_t stamp = est > 0 ? uint64_t(std::llround(est)) : 0;
    std::vector<uint8_t> payload;
    size_t k = 0;
    while (k < records.size())
    {
        size_t end = k, bytes = 0;
        while (end < records.size() &&
               (end == k || bytes + records[end].size() <= kMaxMessagePayload - kMaxOwnerHeader))
            bytes += records[end++].size();
        payload.clear();
        writeVarint(payload, stamp);
        writeVarint(payload, seq);
        writeVarint(payload, end - k);
        for (; k < end; ++k) payload.insert(payload.end(), records[k].begin(), records[k].end());
        w.sendRaw(0, MessageId(EngineMessage::OwnerState), payload.data(), payload.size());
    }
}

void Replication::onOwnerState(uint8_t from, const uint8_t* data, size_t size)
{
    if (!w.isHost() || from == 0)
    {
        w.rejectInput(from, "OwnerState: only clients send it to the host");
        return;
    }
    Registry& reg = w.reg;
    ByteReader r(data, size);
    const uint64_t stamp = r.varint();
    const uint64_t seq   = r.varint();
    const uint64_t n     = r.varint();
    if (!r.ok || n > size)
    {
        w.rejectInput(from, "OwnerState: a malformed header");
        return;
    }
    // The sequence counts as received only if no record was dropped as stale (finishNetwork waits for it)
    bool stale = false;
    EntityRecord rec;
    for (uint64_t i = 0; i < n; ++i)
    {
        if (!readRecord(r, rec))
        {
            w.rejectInput(from, "OwnerState: a malformed record");
            return;
        }
        const Entity e = entityOfNetId(rec.netId);
        if (!e.valid()) continue;  // destroyed meanwhile
        if (w.ownerOf(e) != from)
        {
            counts.ownerStatesDropped++;
            continue;
        }
        Entity root = e;
        while (const Parent* p = reg.tryRead<Parent>(root)) root = p->entity;
        const HostSlot* rs = hostSlot(root);
        // Serial-number compare: a state from before the last OwnerCommand is dropped
        if (!rs || int8_t(uint8_t(rec.counter - rs->command)) < 0)
        {
            counts.ownerStatesDropped++;
            stale = true;
            continue;
        }
        OwnerRecord& o = ownerRecords[e.index];
        if (o.generation != e.generation || o.peer != from)
        {
            o            = OwnerRecord{};
            o.generation = e.generation;
            o.peer       = from;
        }
        for (const ComponentBytes& c : rec.components)
        {
            // Every ComponentId is below kMaxComponents (the id type has no larger values)
            static_assert(size_t(std::numeric_limits<ComponentId>::max()) < kMaxComponents);
            if (!reg.registered(c.id) || reg.componentInfo(c.id).replicate != Replicate::OwnerAuthoritative) continue;
            if (c.id == Transform::id)
            {
                if (root != e) continue;
                Transform t = reg.read<Transform>(e);
                NetReader nr(c.data, c.size, this);
                t.visit(nr);
                if (!nr.ok()) continue;
                if (o.havePose)
                {
                    o.havePrevious     = true;
                    o.previousStamp    = o.stamp;
                    o.previousPosition = o.position;
                }
                o.havePose = true;
                o.stamp    = double(stamp);
                o.position = t.position();
                o.rotation = t.rotation();
                o.scale    = t.scale();
                // The host renders a client-owned entity from the interpolation buffer (§13.3)
                auto& b = buffers[e.index];
                if (b.first != e.generation)
                {
                    b.first = e.generation;
                    b.second.clear();
                }
                b.second.push(double(stamp), t.position(), t.rotation(), rec.counter);
                if (!reg.has<RenderPose>(e)) w.add<RenderPose>(e, RenderPose{t.position(), t.rotation(), 0});
                continue;
            }
            bool replaced = false;
            for (auto& x : o.components)
                if (x.first == c.id)
                {
                    x.second.assign(c.data, c.data + c.size);
                    replaced = true;
                }
            if (!replaced) o.components.push_back({c.id, std::vector<uint8_t>(c.data, c.data + c.size)});
        }
    }
    if (!stale) ownerSeq[from] = std::max(ownerSeq[from], seq);
}

void Replication::ownerPose(bool extrapolate)
{
    Registry& reg = w.reg;
    for (auto it = ownerRecords.begin(); it != ownerRecords.end();)
    {
        const Entity e = reg.aliveAt(it->first);
        OwnerRecord& o = it->second;
        if (!e.valid() || e.generation != o.generation || w.ownerOf(e) != o.peer)
        {
            it = ownerRecords.erase(it);
            continue;
        }
        for (const auto& c : o.components)
        {
            if (!reg.entityMask(e).test(c.first)) continue;
            checkWritten(e, c.first, "owner pose receiver");
            NetReader r(c.second.data(), c.second.size(), this);
            reg.poolById(c.first)->netReadInto(e.index, r);
            noteWritten(e, c.first);
        }
        if (o.havePose && !reg.entityMask(e).test(Parent::id))
        {
            Vec3 p = o.position;
            // Extrapolated from the last two poses to the current host step, at most 6 steps (§13.4); not when the
            // stamps are equal or not increasing (asynchronous stepping, clock jumps)
            if (extrapolate && o.havePrevious && o.stamp > o.previousStamp)
            {
                const double k = std::min(6.0, double(w.steps) - o.stamp);
                if (k > 0) p += (o.position - o.previousPosition) * (k / (o.stamp - o.previousStamp));
            }
            checkWritten(e, Transform::id, "owner pose receiver");
            Transform& t = reg.get<Transform>(e);
            t.setPosition(p);
            t.setRotation(o.rotation);
            t.setScale(o.scale);
            noteWritten(e, Transform::id);
        }
        ++it;
    }
}

void ownerPoseSystem(World& world, double)
{
    world.replication().ownerPose(true);
}

void Replication::extract(bool networkOnly)
{
    if (w.isHost())
    {
        hostReplicate(networkOnly);
        if (w.debugAfterReplication) w.debugAfterReplication();
    }
    else
        ownerStateSend(networkOnly);
    if (!networkOnly) interpolationSystem(w, w.stepDt);
}

// ---- client handlers ----------------------------------------------------------------------------------------------

bool Replication::applyComponent(Entity e, ComponentId id, const uint8_t* data, size_t size, bool rememberUnresolved,
                                 NetId netId)
{
    Registry& reg = w.reg;
    if (id == Transform::id)
    {
        applyTransform(e, data, size, false, false);
        return true;
    }
    if (!reg.registered(id) || id == Parent::id || id == Behaviour::id) return id != Parent::id || true;
    NetReader r(data, size, this);
    if (id == Persistent::id)
    {
        // Through the index path (§9.6); a Persistent never changes
        if (reg.entityMask(e).test(id)) return true;
        Persistent p;
        r.read(p);
        if (!r.ok() || !p.persistentId.valid() || w.persistentIndex.count(p.persistentId.value)) return false;
        w.addNow<Persistent>(e, std::move(p));
        return true;
    }
    const bool added = !reg.entityMask(e).test(id);
    // RenderInstance: the name before the write (an unchanged name appends no record; S19b)
    std::string previousMesh;
    if (id == RenderInstance::id && !added) previousMesh = reg.read<RenderInstance>(e).mesh;
    if (added) w.addDefaultInternal(e, id);
    if (!reg.poolById(id)->netReadInto(e.index, r)) return false;
    if (id == NetOwner::id) w.markOwnersDirty();
    // Every RenderInstance name write goes through renderMeshWritten (an unknown name is logged and kept, not drawn)
    if (id == RenderInstance::id) w.renderMeshWritten(e, added, &previousMesh);
    if (r.unresolved() && rememberUnresolved)
    {
        for (PendingDecode& p : pendingDecodes)
            if (p.netId == netId && p.component == id)
            {
                p.bytes.assign(data, data + size);
                return true;
            }
        pendingDecodes.push_back(PendingDecode{netId, id, std::vector<uint8_t>(data, data + size)});
    }
    return true;
}

void Replication::applyTransform(Entity e, const uint8_t* data, size_t size, bool teleport, bool track)
{
    Transform decoded = w.reg.read<Transform>(e);
    NetReader r(data, size, this);
    decoded.visit(r);
    if (!r.ok()) return;
    if (track) checkWritten(e, Transform::id, "snapshot receiver");
    Transform& t = w.reg.get<Transform>(e);
    if (teleport)
        t.teleport(decoded.position(), decoded.rotation());
    else
    {
        t.setPosition(decoded.position());
        t.setRotation(decoded.rotation());
    }
    t.setScale(decoded.scale());
    noteWritten(e, Transform::id);
}

void Replication::retryPending()
{
    for (size_t i = 0; i < pendingDecodes.size();)
    {
        const Entity e = entityOfNetId(pendingDecodes[i].netId);
        if (!e.valid())
        {
            pendingDecodes.erase(pendingDecodes.begin() + std::ptrdiff_t(i));
            continue;
        }
        const ComponentId id = pendingDecodes[i].component;
        NetReader r(pendingDecodes[i].bytes.data(), pendingDecodes[i].bytes.size(), this);
        if (w.reg.entityMask(e).test(id) && w.reg.poolById(id)->netReadInto(e.index, r)) noteWritten(e, id);
        if (r.ok() && !r.unresolved())
            pendingDecodes.erase(pendingDecodes.begin() + std::ptrdiff_t(i));
        else
            ++i;
    }
}

void Replication::onSpawn(uint8_t from, const uint8_t* data, size_t size)
{
    if (w.isHost() || from != 0)
    {
        w.rejectInput(from, "Spawn: only the host sends it to clients");
        return;
    }
    Registry& reg = w.reg;
    ByteReader r(data, size);
    const uint64_t id64   = r.varint();
    const ClassId classId = r.u16();
    const uint8_t owner   = r.u8();
    const uint64_t parent = r.varint();
    const uint8_t snap    = r.u8();
    const uint8_t command = r.u8();
    const uint64_t localN = r.varint();
    const uint8_t* localP = r.bytes(localN);
    std::vector<ComponentBytes> comps;
    const bool compsOk     = r.ok && readComponents(r, comps);
    const uint64_t present = r.varint();
    std::vector<ComponentId> presence;
    for (uint64_t i = 0; i < present && r.ok && i < 256; ++i) presence.push_back(r.u8());
    if (!compsOk || !r.atEnd() || id64 == 0 || id64 > 0xFFFFFFFFull || parent > 0xFFFFFFFFull || present > 256)
    {
        w.rejectInput(from, "Spawn: malformed");
        return;
    }
    const NetId netId = NetId(id64);
    Immediate scope(w.flushing);
    Entity e;
    const bool bound = netId < kFirstDynamicNetId;
    if (bound)
    {
        if (netId == kWorldStateNetId)
            e = w.worldState;
        else if (netId >= kFirstSpaceStateNetId && netId - kFirstSpaceStateNetId < w.spaceStates.size())
            e = w.spaceStates[netId - kFirstSpaceStateNetId];
        // Every build: the host has state components this client does not register
        if (!e.valid())
            SAIGA_EXIT_ERROR("Spawn: the bound NetId " + std::to_string(netId) +
                             " names a world-state or space-state entity this client does not have (a setup mismatch)");
        if (netIdOf(e) != netId) mapClient(netId, e, snap, command);
    }
    else
    {
        e = entityOfNetId(netId);
        if (!e.valid())
        {
            e = w.create();
            mapClient(netId, e, snap, command);
        }
    }
    for (const ComponentBytes& c : comps)
    {
        if (c.id == Transform::id)
        {
            if (!bound) applyTransform(e, c.data, c.size, true, false);
            continue;
        }
        if (!applyComponent(e, c.id, c.data, c.size, true, netId))
            w.rejectInput(from, "Spawn: the component " + std::to_string(int(c.id)) + " does not decode");
    }
    for (ComponentId c : presence)
        if (presenceComponent(reg, c) && !reg.entityMask(e).test(c)) w.addDefaultInternal(e, c);
    // Ownership is defined on roots (the header carries the root's owner for children too)
    if (owner != 0 && parent == 0 && !reg.entityMask(e).test(NetOwner::id))
    {
        w.addDefaultInternal(e, NetOwner::id);
        reg.get<NetOwner>(e).peer = owner;
        w.markOwnersDirty();
    }
    if (parent != 0 && !reg.entityMask(e).test(Parent::id))
    {
        if (localP && localN > 0)
        {
            LocalTransform local;
            NetReader lr(localP, size_t(localN), this);
            lr.read(local);
            if (lr.ok())
            {
                if (reg.entityMask(e).test(LocalTransform::id))
                    reg.get<LocalTransform>(e) = local;
                else
                    w.addNow<LocalTransform>(e, std::move(local));
            }
        }
        const Entity p = entityOfNetId(NetId(parent));
        if (p.valid())
            w.addNow<Parent>(e, Parent{p});
        else
            pendingParents[netId] = NetId(parent);  // waits for the parent instead of destroying the child
    }
    // Remote entities are rendered from the interpolation buffer
    if (!bound && owner != w.selfPeer && !reg.entityMask(e).test(RenderPose::id))
    {
        w.addDefaultInternal(e, RenderPose::id);
        RenderPose& rp = reg.get<RenderPose>(e);
        rp.position    = reg.read<Transform>(e).position();
        rp.rotation    = reg.read<Transform>(e).rotation();
        rp.flags       = RenderPose::kChanged | RenderPose::kTeleported;
    }
    for (size_t c = 0; c < kMaxComponents; ++c)
        if (reg.entityMask(e).test(ComponentId(c)) && replicatedMode(reg.componentInfo(ComponentId(c)).replicate))
            noteWritten(e, ComponentId(c));
    // Children that waited for this entity
    for (auto it = pendingParents.begin(); it != pendingParents.end();)
    {
        if (it->second != netId)
        {
            ++it;
            continue;
        }
        const Entity child = entityOfNetId(it->first);
        if (child.valid() && !reg.entityMask(child).test(Parent::id)) w.addNow<Parent>(child, Parent{e});
        it = pendingParents.erase(it);
    }
    retryPending();
    // The game object: configure and onReplicatedSpawn, never onSpawn (§13.5)
    if (classId != 0 && !reg.entityMask(e).test(Behaviour::id))
    {
        const ClassList::ClassInfo* info = w.classes.find(classId);
        if (!info)
        {
            w.rejectInput(from, "Spawn: unknown class " + std::to_string(classId));
            return;
        }
        Behaviour b;
        b.object        = info->factory();
        b.object->owner = &w;
        b.object->self  = e;
        b.classId       = info->id;
        w.addNow<Behaviour>(e, std::move(b));
        GameObject* o = reg.get<Behaviour>(e).object.get();
        w.hookDepth++;
        try
        {
            o->configure();
            if (reg.alive(e) && reg.entityMask(e).test(Behaviour::id)) o->onReplicatedSpawn();
        }
        catch (...)
        {
            w.hookDepth--;
            // The object is removed; the entity and its mapping stay (it must stay in sync)
            if (reg.alive(e) && reg.entityMask(e).test(Behaviour::id)) w.removeNow<Behaviour>(e);
            throw;
        }
        w.hookDepth--;
        w.releaseRetired();
    }
}

void Replication::destroyReplicated(Entity e, bool callbacks)
{
    Registry& reg              = w.reg;
    std::vector<Entity> doomed = w.collectDoomed(e);
    if (callbacks)
    {
        // onReplicatedDestroy only (never onDestroy), children first
        std::vector<Entity> hooked;
        w.hookDepth++;
        try
        {
            for (auto it = doomed.rbegin(); it != doomed.rend(); ++it)
            {
                if (!reg.alive(*it)) continue;
                Behaviour* b = reg.tryGet<Behaviour>(*it);
                if (!b || !b->object || b->destroying) continue;
                b->destroying = true;
                hooked.push_back(*it);
                b->object->onReplicatedDestroy();
            }
        }
        catch (...)
        {
            w.hookDepth--;
            // The host dropped the NetId and the Destroy is consumed: the replica is destroyed anyway (it must stay in
            // sync, as the spawn path keeps it), then the exception propagates
            if (reg.alive(e))
            {
                for (Entity x : w.collectDoomed(e))
                {
                    if (!reg.alive(x)) continue;
                    if (Behaviour* bh = reg.tryGet<Behaviour>(x)) bh->destroying = true;
                    unmapClient(x);
                }
                w.destroyOne(e);
                w.releaseRetired();
            }
            throw;
        }
        w.hookDepth--;
        if (!reg.alive(e)) return;
        doomed = w.collectDoomed(e);
    }
    for (Entity x : doomed)
    {
        if (!reg.alive(x)) continue;
        if (Behaviour* b = reg.tryGet<Behaviour>(x)) b->destroying = true;
        unmapClient(x);
    }
    w.destroyOne(e);
    w.releaseRetired();
}

void Replication::onDestroy(uint8_t from, const DestroyMessage& m)
{
    if (w.isHost() || from != 0)
    {
        w.rejectInput(from, "Destroy: only the host sends it to clients");
        return;
    }
    const Entity e = entityOfNetId(m.netId);
    // Already removed by the client's own cascade (a destroyed parent): ignored
    if (!e.valid() || m.netId < kFirstDynamicNetId) return;
    Immediate scope(w.flushing);
    destroyReplicated(e, true);
}

void Replication::onHierarchy(uint8_t from, const HierarchyMessage& m)
{
    if (w.isHost() || from != 0)
    {
        w.rejectInput(from, "Hierarchy: only the host sends it to clients");
        return;
    }
    Registry& reg  = w.reg;
    const Entity e = entityOfNetId(m.netId);
    if (!e.valid()) return;
    Immediate scope(w.flushing);
    if (reg.entityMask(e).test(Parent::id)) w.removeNow<Parent>(e);
    if (reg.entityMask(e).test(LocalTransform::id))
    {
        checkWritten(e, LocalTransform::id, "hierarchy receiver");
        reg.get<LocalTransform>(e) = m.local;
    }
    else
        w.addNow<LocalTransform>(e, LocalTransform(m.local));
    noteWritten(e, LocalTransform::id);
    pendingParents.erase(m.netId);
    if (m.parent == 0) return;
    const Entity p = entityOfNetId(m.parent);
    if (p.valid())
        w.addNow<Parent>(e, Parent{p});
    else
        pendingParents[m.netId] = m.parent;
}

void Replication::onComponentChange(uint8_t from, const uint8_t* data, size_t size)
{
    if (w.isHost() || from != 0)
    {
        w.rejectInput(from, "ComponentChange: only the host sends it to clients");
        return;
    }
    Registry& reg = w.reg;
    ByteReader r(data, size);
    const uint64_t id   = r.varint();
    const uint8_t kind  = r.u8();
    const ComponentId c = r.u8();
    if (!r.ok || id == 0 || id > 0xFFFFFFFFull || kind > kChangePresenceRemove || !reg.registered(c) ||
        c == Transform::id || c == Parent::id)
    {
        w.rejectInput(from, "ComponentChange: malformed");
        return;
    }
    const Entity e = entityOfNetId(NetId(id));
    if (!e.valid()) return;
    Immediate scope(w.flushing);
    const bool has = reg.entityMask(e).test(c);
    if (kind == kChangeSet)
    {
        if (has) checkWritten(e, c, "component change receiver");
        if (!applyComponent(e, c, data + r.pos, size - r.pos, true, NetId(id)))
            w.rejectInput(from, "ComponentChange: the component does not decode");
        else
            noteWritten(e, c);
    }
    else if (kind == kChangePresenceAdd)
    {
        if (!has && presenceComponent(reg, c)) w.addDefaultInternal(e, c);
    }
    else if (has && !(reg.flags(e) & (Registry::kWorldState | Registry::kSpaceState) &&
                      (reg.isGlobal(c) || reg.isSpaceState(c))))
    {
        w.removeInternal(e, c);
        written.erase(componentKey(e.index, c));
    }
}

void Replication::onSnapshot(uint8_t from, const uint8_t* data, size_t size)
{
    if (w.isHost() || from != 0)
    {
        w.rejectInput(from, "Snapshot: only the host sends it to clients");
        return;
    }
    Registry& reg = w.reg;
    ByteReader r(data, size);
    const uint8_t flags  = r.u8();
    const uint64_t epoch = r.varint();
    const uint64_t step  = r.varint();
    const uint64_t seq   = r.varint();
    const uint64_t part  = r.varint();
    const uint64_t parts = r.varint();
    if (!r.ok || parts == 0 || part >= parts || parts > 65536 || flags > kSnapshotCompressed)
    {
        w.rejectInput(from, "Snapshot: a malformed header");
        return;
    }
    if (epoch != currentEpoch)
    {
        // Sent before a host load (§13.4)
        counts.snapshotsDroppedEpoch++;
        return;
    }
    if (seq <= snapshotSeq) return;
    std::vector<uint8_t> body;
    if (flags & kSnapshotCompressed)
    {
        const unsigned long long n = ZSTD_getFrameContentSize(data + r.pos, size - r.pos);
        if (n == ZSTD_CONTENTSIZE_ERROR || n == ZSTD_CONTENTSIZE_UNKNOWN || n > kMaxDecompressed)
        {
            w.rejectInput(from, "Snapshot: a malformed compressed body");
            return;
        }
        body.resize(size_t(n));
        const size_t got = ZSTD_decompress(body.data(), body.size(), data + r.pos, size - r.pos);
        if (ZSTD_isError(got) || got != body.size())
        {
            w.rejectInput(from, "Snapshot: a malformed compressed body");
            return;
        }
    }
    else
        body.assign(data + r.pos, data + size);
    // Every record must parse before anything is used
    {
        ByteReader check(body.data(), body.size());
        EntityRecord rec;
        while (check.ok && check.pos < check.n)
            if (!readRecord(check, rec)) check.ok = false;
        if (!check.ok)
        {
            w.rejectInput(from, "Snapshot: a malformed record");
            return;
        }
    }
    // Every snapshot of the batch goes to the interpolation buffers (remote roots)
    {
        ByteReader br(body.data(), body.size());
        EntityRecord rec;
        while (br.pos < br.n && readRecord(br, rec))
        {
            const Entity e = entityOfNetId(rec.netId);
            if (!e.valid() || rec.netId < kFirstDynamicNetId || w.ownerOf(e) == w.selfPeer ||
                reg.entityMask(e).test(Parent::id))
                continue;
            for (const ComponentBytes& c : rec.components)
            {
                if (c.id != Transform::id) continue;
                Transform t = reg.read<Transform>(e);
                NetReader nr(c.data, c.size, this);
                t.visit(nr);
                if (!nr.ok()) break;
                auto& b = buffers[e.index];
                if (b.first != e.generation)
                {
                    b.first = e.generation;
                    b.second.clear();
                }
                b.second.push(double(step), t.position(), t.rotation(), rec.counter);
            }
        }
    }
    hostClock.observeSnapshot(step, w.steps);
    SnapshotParts& sp = pendingSnapshots[seq];
    if (sp.parts == 0)
    {
        sp.hostStep = step;
        sp.parts    = uint32_t(parts);
        sp.bodies.resize(size_t(parts));
    }
    if (sp.parts != parts || !sp.bodies[part].empty() || body.empty()) return;
    sp.bodies[part] = std::move(body);
    if (++sp.received == sp.parts) newestComplete = std::max(newestComplete, seq);
}

void Replication::applySnapshot(const SnapshotParts& s)
{
    Registry& reg = w.reg;
    EntityRecord rec;
    for (const std::vector<uint8_t>& body : s.bodies)
    {
        ByteReader br(body.data(), body.size());
        while (br.pos < br.n && readRecord(br, rec))
        {
            const Entity e = entityOfNetId(rec.netId);
            if (!e.valid()) continue;  // the reliable spawn follows
            const bool owned   = w.ownerOf(e) == w.selfPeer;
            ClientEntry* ce    = clientEntry(e);
            const bool snapped = ce && !owned && rec.counter != ce->snap;
            for (const ComponentBytes& c : rec.components)
            {
                if (!reg.registered(c.id)) continue;
                const Replicate mode = reg.componentInfo(c.id).replicate;
                // The owner keeps its own owner-authoritative state
                if (owned && mode == Replicate::OwnerAuthoritative) continue;
                if (c.id == Transform::id)
                {
                    if (rec.netId < kFirstDynamicNetId || reg.entityMask(e).test(Parent::id)) continue;
                    applyTransform(e, c.data, c.size, snapped, true);
                    continue;
                }
                if (c.id == Parent::id || c.id == Persistent::id || c.id == NetOwner::id ||
                    !reg.entityMask(e).test(c.id))
                    continue;
                checkWritten(e, c.id, "snapshot receiver");
                NetReader nr(c.data, c.size, this);
                reg.poolById(c.id)->netReadInto(e.index, nr);
                noteWritten(e, c.id);
            }
            // The owner ignores the snapshot counter (its teleports come as OwnerCommands)
            if (ce && !owned) ce->snap = rec.counter;
        }
    }
}

void Replication::afterReceive(bool networkOnly)
{
    if (w.isHost()) return;
    if (newestComplete > snapshotSeq)
    {
        auto it = pendingSnapshots.find(newestComplete);
        if (it != pendingSnapshots.end()) applySnapshot(it->second);
        snapshotSeq = newestComplete;
        counts.snapshotsApplied++;
    }
    // Older snapshots are never applied; incomplete ones far behind are dropped
    while (!pendingSnapshots.empty() &&
           (pendingSnapshots.begin()->first <= snapshotSeq || pendingSnapshots.size() > 64))
        pendingSnapshots.erase(pendingSnapshots.begin());
    if (!networkOnly) hostClock.update();
    w.hostStepOffset = hostClock.valid() ? hostClock.offset() : 0;
}

void Replication::onOwnerCommand(uint8_t from, const OwnerCommandMessage& m)
{
    if (w.isHost() || from != 0)
    {
        w.rejectInput(from, "OwnerCommand: only the host sends it to owners");
        return;
    }
    const Entity e = entityOfNetId(m.netId);
    if (!e.valid() || w.ownerOf(e) != w.selfPeer) return;
    if (m.kind == OwnerCommandMessage::kTeleport)
    {
        w.reg.get<Transform>(e).teleport(m.position, m.rotation);
        noteWritten(e, Transform::id);
    }
    if (ClientEntry* ce = clientEntry(e)) ce->command = m.counter;
}

void Replication::onPlayerAction(uint8_t from, const PlayerActionMessage& m)
{
    if (!w.isHost() || from == 0)
    {
        w.rejectInput(from, "PlayerAction: only clients send it to the host");
        return;
    }
    const Entity e = entityOfNetId(m.player);
    // Only for players the sender owns (dropped and counted otherwise)
    if (!e.valid() || !w.reg.entityMask(e).test(Player::id) || w.ownerOf(e) != from)
    {
        counts.actionsDropped++;
        return;
    }
    // Into the action queue with (sender, sequence): an ActionEvent in the next step, exactly once
    w.actions.push_back(World::QueuedAction{from, w.actionSequence++, PlayerAction{e, m.action, m.aim}});
}

void Replication::onPing(uint8_t from, const PingMessage& m)
{
    if (!w.isHost() || from == 0)
    {
        w.rejectInput(from, "Ping: only clients send it to the host");
        return;
    }
    w.send(from, PongMessage{m.clientStep, w.steps});
}

void Replication::onPong(uint8_t from, const PongMessage& m)
{
    if (w.isHost() || from != 0)
    {
        w.rejectInput(from, "Pong: only the host sends it to clients");
        return;
    }
    hostClock.observePong(m.clientStep, m.hostStep, w.steps);
}

void Replication::onResync(uint8_t from, const ResyncMessage& m)
{
    if (w.isHost() || from != 0)
    {
        w.rejectInput(from, "Resync: only the host sends it to clients");
        return;
    }
    Immediate scope(w.flushing);
    currentEpoch = m.epoch;
    // Every replicated entity goes without callbacks; the bound state entities stay (the spawns overwrite them)
    std::vector<Entity> all;
    for (const auto& kv : clientByNetId)
        if (kv.first >= kFirstDynamicNetId && w.reg.alive(kv.second.entity)) all.push_back(kv.second.entity);
    std::sort(all.begin(), all.end());
    for (Entity e : all)
        if (w.reg.alive(e)) destroyReplicated(e, false);
    clientByNetId.clear();
    clientNetIdOf.clear();
    pendingParents.clear();
    pendingDecodes.clear();
    pendingSnapshots.clear();
    buffers.clear();
    written.clear();
}

// ---- the write detector -------------------------------------------------------------------------------------------

void Replication::setWriteDetector(std::function<void(const std::string&)> report)
{
    detector = std::move(report);
    written.clear();
}

uint64_t Replication::currentHash(Entity e, ComponentId id) const
{
    std::vector<uint8_t> bytes;
    if (id == Transform::id)
    {
        BinaryWriter bw(bytes);
        bw.write(w.reg.read<Transform>(e));
    }
    else
        w.reg.poolById(id)->appendBytes(e.index, bytes);
    return fnv1a64Bytes(bytes.data(), bytes.size());
}

void Replication::checkWritten(Entity e, ComponentId id, const char* receiver)
{
    if (!detector) return;
    auto it = written.find(componentKey(e.index, id));
    if (it == written.end() || it->second.first != e.generation) return;
    if (it->second.second != currentHash(e, id))
        detector("peer " + std::to_string(w.selfPeer) + ": " + std::string(w.reg.componentInfo(id).name) + " of " +
                 describe(e) + " was written by someone else than the " + receiver +
                 " (it belongs to a remote authority, R-NET-2)");
}

void Replication::noteWritten(Entity e, ComponentId id)
{
    if (!detector || !w.reg.alive(e) || !w.reg.entityMask(e).test(id)) return;
    written[componentKey(e.index, id)] = {e.generation, currentHash(e, id)};
}

}  // namespace Engine
}  // namespace Saiga
