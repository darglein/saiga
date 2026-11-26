/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The world over the network (docs/engine_design.md §13.2, §13.6; S26): the network schema, the handshake (JoinRequest,
// JoinReject, JoinAccept, Resync v2, admission), chunk interest, snapshots and indexes, the client's per-chunk queue,
// re-requests, unloads and the host's subscriptions, over LoopbackTransport.

#include "saiga/engine/content/ChunkRecords.h"
#include "saiga/engine/content/GenerationPipeline.h"
#include "saiga/engine/core/Log.h"
#include "saiga/engine/gameplay/GameObject.h"
#include "saiga/engine/net/ChunkMessages.h"
#include "saiga/engine/net/EngineMessages.h"
#include "saiga/engine/net/JoinCheck.h"
#include "saiga/engine/net/JoinMessages.h"
#include "saiga/engine/net/NetSchema.h"
#include "saiga/engine/net/Varint.h"
#include "saiga/engine/streaming/ChunkResidency.h"
#include "saiga/engine/terrain/HookGenerator.h"
#include "saiga/engine/terrain/TerrainDeltaCodec.h"
#include "saiga/engine/terrain/TerrainMessages.h"
#include "saiga/engine/world/Events.h"
#include "saiga/engine/world/World.h"

#include "gtest/gtest.h"

#include <atomic>
#include <chrono>
#include <iostream>
#include <string>
#include <thread>

#include "world_net_test_setup.h"

using namespace Saiga;
using namespace Saiga::Engine;

namespace
{
const bool g_noEcho = (setLogEcho(false), true);

// ---- NetSchema ------------------------------------------------------------------------------------------------------

// The registration variants of HashCoversRegistration (selected by g_variant before a world is built)
int g_variant = 0;
enum Variant
{
    Base,
    FieldAdded,
    ComponentVersion,
    ReplicateMode,
    Environment,
    EventAdded,
    ClassAdded,
    MessageVersion,
    MessageChannel,
    ActionAdded,
    HeldAdded,
    TemplateSpacing,
    Streaming,
    NotSaved,
    StreamingRadius,
    VariantCount
};

struct SchemaComp
{
    static constexpr const char* name = "SchemaComp";
    static constexpr ComponentId id   = kFirstGameComponent;
    static constexpr uint32_t version = 1;
    int32_t a                         = 0;
    template <class V>
    void visit(V& v)
    {
        v("a", a);
    }
};
struct SchemaCompField
{
    static constexpr const char* name = "SchemaComp";
    static constexpr ComponentId id   = kFirstGameComponent;
    static constexpr uint32_t version = 1;
    int32_t a = 0, b = 0;
    template <class V>
    void visit(V& v)
    {
        v("a", a);
        v("b", b);
    }
};
struct SchemaCompV2
{
    static constexpr const char* name = "SchemaComp";
    static constexpr ComponentId id   = kFirstGameComponent;
    static constexpr uint32_t version = 2;
    int32_t a                         = 0;
    template <class V>
    void visit(V& v)
    {
        v("a", a);
    }
};
struct SchemaEvent
{
    static constexpr const char* name = "SchemaEvent";
    static constexpr EventId id       = 64;
    static constexpr uint32_t version = 1;
    int32_t value                     = 0;
    template <class V>
    void visit(V& v)
    {
        v("value", value);
    }
};
class SchemaObject final : public GameObject
{
   public:
    static constexpr ClassId id       = kFirstGameClass;
    static constexpr const char* name = "SchemaObject";
};
struct SchemaMessage
{
    static constexpr MessageId id     = 64;
    static constexpr const char* name = "SchemaMessage";
    static constexpr uint32_t version = 1;
    int32_t v                         = 0;
    template <class V>
    void visit(V& vis)
    {
        vis("v", v);
    }
};
struct SchemaMessageV2
{
    static constexpr MessageId id     = 64;
    static constexpr const char* name = "SchemaMessage";
    static constexpr uint32_t version = 2;
    int32_t v                         = 0;
    template <class V>
    void visit(V& vis)
    {
        vis("v", v);
    }
};
struct SchemaMessageState
{
    static constexpr MessageId id     = 64;
    static constexpr const char* name = "SchemaMessage";
    static constexpr uint32_t version = 1;
    static constexpr Channel channel  = Channel::State;
    int32_t v                         = 0;
    template <class V>
    void visit(V& vis)
    {
        vis("v", v);
    }
};

void schemaComponents(Registry& r)
{
    if (g_variant == FieldAdded)
        r.registerComponent<SchemaCompField>(Replicate::Host);
    else if (g_variant == ComponentVersion)
        r.registerComponent<SchemaCompV2>(Replicate::Host);
    else if (g_variant == ReplicateMode)
        r.registerComponent<SchemaComp>(Replicate::HostReliable);
    else if (g_variant == NotSaved)
        r.registerComponent<SchemaComp>(Replicate::Host, Saved::No);
    else
        r.registerComponent<SchemaComp>(Replicate::Host);
}
void schemaEvents(EventRegistry& e)
{
    if (g_variant == EventAdded) e.add<SchemaEvent>();
}
void schemaClasses(ClassList& c)
{
    if (g_variant == ClassAdded) c.add<SchemaObject>();
}
void onSchemaMessage(World&, uint8_t, const SchemaMessage&) {}
void onSchemaMessageV2(World&, uint8_t, const SchemaMessageV2&) {}
void onSchemaMessageState(World&, uint8_t, const SchemaMessageState&) {}
void schemaMessages(MessageList& l)
{
    if (g_variant == MessageVersion)
        l.add<SchemaMessageV2>(Reliability::Reliable, onSchemaMessageV2);
    else if (g_variant == MessageChannel)
        l.add<SchemaMessageState>(Reliability::Reliable, onSchemaMessageState);
    else
        l.add<SchemaMessage>(Reliability::Reliable, onSchemaMessage);
}
void schemaActions(ActionList& a)
{
    a.add(1, "jump");
    a.addHeld(0, "fire");
    if (g_variant == ActionAdded) a.add(2, "crouch");
    if (g_variant == HeldAdded) a.addHeld(1, "aim");
}

WorldSetup schemaSetup(int variant)
{
    g_variant = variant;
    WorldSetup s;
    s.profiling  = false;
    s.components = schemaComponents;
    s.events     = schemaEvents;
    s.classes    = schemaClasses;
    s.messages   = schemaMessages;
    s.actions    = schemaActions;
    SpaceTemplate t;
    t.name       = "flat";
    t.sizeChunks = 16;
    if (variant == TemplateSpacing) t.baseSpacing = 1;
    if (variant == StreamingRadius) t.fullRadius = 100;
    s.config.spaceTemplates.push_back(t);
    s.config.streaming    = variant == Streaming;
    s.environment.enabled = variant == Environment;
    return s;
}

TEST(NetSchema, EqualSetupsHashEqual)
{
    World a(schemaSetup(Base)), b(schemaSetup(Base));
    EXPECT_EQ(a.netSchemaHash(), b.netSchemaHash());
    EXPECT_EQ(a.netSchema().sectionHashes(), b.netSchema().sectionHashes());
    EXPECT_EQ(a.netSchemaHash(), a.netSchema().hash());
    std::vector<uint8_t> ea, eb;
    a.netSchema().encode(ea);
    b.netSchema().encode(eb);
    EXPECT_EQ(ea, eb);
    // Every registered component is in it (also Saved::No ones), the messages with their versions
    bool comp = false;
    for (const NetSchema::Type& t : a.netSchema().types) comp = comp || t.name == "SchemaComp";
    EXPECT_TRUE(comp);
    EXPECT_FALSE(a.netSchema().messages.empty());
}

TEST(NetSchema, HashCoversRegistration)
{
    World base(schemaSetup(Base));
    const auto baseSections = base.netSchema().sectionHashes();
    // The section that must differ per variant
    const std::vector<std::pair<int, NetSchemaSection>> changes = {
        {FieldAdded, NetSchemaSection::Types},          {ComponentVersion, NetSchemaSection::Types},
        {ReplicateMode, NetSchemaSection::Types},       {Environment, NetSchemaSection::Types},
        {EventAdded, NetSchemaSection::Types},          {ClassAdded, NetSchemaSection::Classes},
        {MessageVersion, NetSchemaSection::Messages},   {MessageChannel, NetSchemaSection::Messages},
        {ActionAdded, NetSchemaSection::Actions},       {HeldAdded, NetSchemaSection::Actions},
        {TemplateSpacing, NetSchemaSection::Templates}, {Streaming, NetSchemaSection::Flags}};
    for (const auto& [variant, section] : changes)
    {
        World w(schemaSetup(variant));
        EXPECT_NE(w.netSchemaHash(), base.netSchemaHash()) << "variant " << variant;
        const auto sections = w.netSchema().sectionHashes();
        // Streaming also registers its engine components (the types change too)
        for (size_t i = 0; i < kNetSchemaSections; ++i)
        {
            if (variant == Streaming && i != size_t(section)) continue;
            EXPECT_EQ(sections[i] != baseSections[i], i == size_t(section))
                << "variant " << variant << ", section " << netSchemaSectionName(NetSchemaSection(i));
        }
    }
    // Not in the schema: Saved, the streaming radii (data files are not read by it either)
    for (int variant : {int(NotSaved), int(StreamingRadius)})
    {
        World w(schemaSetup(variant));
        EXPECT_EQ(w.netSchemaHash(), base.netSchemaHash()) << "variant " << variant;
    }
}

// ---- helpers of the network tests
// ------------------------------------------------------------------------------------

using namespace WorldNetTest;

// The chunk (4, 5) of the overworld (site 0 is in it) and its east neighbour (5, 5)
const ChunkId kC = chunkAt(0, 4, 5);
const ChunkId kD = chunkAt(0, 5, 5);

Vec3 centreOf(const ChunkId& c)
{
    return Vec3(double(c.x) * 256 + 128, 0, double(c.z) * 256 + 128);
}

// No pending chunk or queued part on the clients (every space indexed without streaming), no queued snapshot
bool synced(NetPeers& n)
{
    if (n.host->replication().subscriptions().queuedSnapshots() != 0) return false;
    for (auto& c : n.clients)
    {
        const ClientChunkSync& s = c->replication().chunkSync();
        if (c->joinState() != JoinState::Joined || s.pendingCount() != 0 || s.queuedParts() != 0) return false;
        if (!c->streaming() && !s.allIndexed()) return false;
    }
    return true;
}

// The n-th tombstone candidate of a chunk that the host accepts
InstanceKey acceptedKey(const World& host, const ChunkId& c, size_t index = 0)
{
    const auto p = host.generation().chunkPlacement(c, host.effectivePlacementVersion(c));
    size_t found = 0;
    for (const TierOneInstance& t : p->instances)
        if (host.checkTombstone(c, t.key, nullptr) && !host.isTombstoned(c, t.key) && found++ == index) return t.key;
    ADD_FAILURE() << "no accepted key in the chunk";
    return InstanceKey();
}

// The client's state of the chunk equals the host's: sequence, dense delta, effective record
void expectChunkEqual(const World& host, const World& client, const ChunkId& c)
{
    EXPECT_EQ(client.terrainEdits().sequence(c), host.terrainEdits().sequence(c)) << c.x << ", " << c.z;
    std::vector<int32_t> a, b;
    host.terrainEdits().denseDelta(c, a);
    client.terrainEdits().denseDelta(c, b);
    EXPECT_EQ(a, b) << c.x << ", " << c.z;
    EXPECT_EQ(client.effectiveRecord(c), host.effectiveRecord(c)) << c.x << ", " << c.z;
}

TerrainPart partOf(const ChunkId& c, uint32_t sequence, int x0, int z0, int w, int h, int32_t value)
{
    TerrainPart p;
    p.chunk    = c;
    p.sequence = sequence;
    p.x0       = x0;
    p.z0       = z0;
    p.w        = w;
    p.h        = h;
    p.deltas.assign(size_t(w) * size_t(h), value);
    return p;
}

std::vector<uint8_t> modificationBytes(uint32_t epoch, SpaceId space, const std::vector<TerrainPart>& parts)
{
    std::vector<uint8_t> out;
    encodeTerrainModification(epoch, space, parts, out);
    return out;
}

// A client-owned player root on the host (Player.peer = owner)
Entity clientPlayer(World& host, const Vec3& p, uint8_t owner = 1, SpaceId space = 0)
{
    const Entity e           = StreamingTest::addPlayer(host, p, space);
    host.get<Player>(e).peer = owner;
    host.setOwner(e, owner);
    return e;
}

// One framed message (S15's packet framing: base sequence, count, delta, ID, length, payload)
std::vector<uint8_t> framed(MessageId id, const std::vector<uint8_t>& payload)
{
    std::vector<uint8_t> p;
    writeVarint(p, 0);
    writeVarint(p, 1);
    writeVarint(p, 0);
    p.push_back(uint8_t(id & 0xFF));
    p.push_back(uint8_t(id >> 8));
    writeVarint(p, payload.size());
    p.insert(p.end(), payload.begin(), payload.end());
    return p;
}

// The messages of the packets a raw loopback end received: (ID, payload)
std::vector<std::pair<MessageId, std::vector<uint8_t>>> receivedMessages(Transport& t)
{
    std::vector<std::pair<MessageId, std::vector<uint8_t>>> out;
    TransportEvent ev;
    while (t.poll(ev))
    {
        if (ev.type != TransportEvent::Packet) continue;
        const uint8_t* d = ev.packet.data();
        const size_t n   = ev.packet.size();
        size_t pos       = 0;
        uint64_t base = 0, count = 0;
        if (!readVarint(d, n, pos, base) || !readVarint(d, n, pos, count)) continue;
        for (uint64_t i = 0; i < count; ++i)
        {
            uint64_t delta = 0, length = 0;
            if (!readVarint(d, n, pos, delta) || pos + 2 > n) break;
            const MessageId id = MessageId(d[pos] | (uint16_t(d[pos + 1]) << 8));
            pos += 2;
            if (!readVarint(d, n, pos, length) || length > n - pos) break;
            out.push_back({id, std::vector<uint8_t>(d + pos, d + pos + length)});
            pos += size_t(length);
        }
    }
    return out;
}

// ---- message formats ------------------------------------------------------------------------------------------------

TEST(JoinMessages, RoundTripsAndBounds)
{
    World w(worldNetTestSetup(false));
    JoinRequest r = joinRequestOf(w);
    std::vector<uint8_t> bytes;
    encodeJoinRequest(r, bytes);
    // The frozen prefix (D3): the protocol version, little-endian
    ASSERT_GE(bytes.size(), 4u);
    EXPECT_EQ((std::vector<uint8_t>(bytes.begin(), bytes.begin() + 4)), (std::vector<uint8_t>{1, 0, 0, 0}));
    JoinRequest d;
    std::string error;
    ASSERT_TRUE(decodeJoinRequest(bytes.data(), bytes.size(), d, error)) << error;
    EXPECT_EQ(d.schemaHash, r.schemaHash);
    EXPECT_EQ(d.sectionHashes, r.sectionHashes);
    EXPECT_EQ(d.renderTableHash, r.renderTableHash);
    EXPECT_EQ(d.seed, r.seed);
    EXPECT_EQ(d.nonLegacy, r.nonLegacy);
    ASSERT_EQ(d.generators.size(), r.generators.size());
    for (size_t i = 0; i < d.generators.size(); ++i)
    {
        EXPECT_EQ(d.generators[i].name, r.generators[i].name);
        EXPECT_EQ(d.generators[i].version, r.generators[i].version);
        EXPECT_EQ(d.generators[i].slots, r.generators[i].slots);
    }
    // Truncation, bytes left over, an unknown flag bit
    for (size_t cut = 1; cut < bytes.size(); cut += 7)
        EXPECT_FALSE(decodeJoinRequest(bytes.data(), cut, d, error)) << cut;
    std::vector<uint8_t> more = bytes;
    more.push_back(0);
    EXPECT_FALSE(decodeJoinRequest(more.data(), more.size(), d, error));
    std::vector<uint8_t> flags = bytes;
    flags[4 + 8 + 1 + 6 * 8 + 8 + 8] |= 0x80;
    EXPECT_FALSE(decodeJoinRequest(flags.data(), flags.size(), d, error));
    EXPECT_NE(error.find("unknown flags"), std::string::npos) << error;
    uint32_t version = 0;
    EXPECT_TRUE(joinRequestVersion(bytes.data(), bytes.size(), version));
    EXPECT_EQ(version, kProtocolVersion);

    // JoinReject: the frozen prefix (reason, text)
    JoinReject reject;
    reject.reason = JoinRejectReason::Schema;
    reject.text   = "ab";
    bytes.clear();
    encodeJoinReject(reject, bytes);
    EXPECT_EQ(bytes, (std::vector<uint8_t>{2, 2, 'a', 'b'}));
    JoinReject rd;
    ASSERT_TRUE(decodeJoinReject(bytes.data(), bytes.size(), rd, error)) << error;
    EXPECT_EQ(rd.reason, JoinRejectReason::Schema);
    EXPECT_EQ(rd.text, "ab");
    bytes.push_back(1);
    EXPECT_FALSE(decodeJoinReject(bytes.data(), bytes.size(), rd, error));
    std::vector<uint8_t> longText{0};
    writeVarint(longText, 5000);
    longText.resize(longText.size() + 5000, 'x');
    EXPECT_FALSE(decodeJoinReject(longText.data(), longText.size(), rd, error));
    const std::vector<uint8_t> unknownReason{99, 0};
    EXPECT_FALSE(decodeJoinReject(unknownReason.data(), unknownReason.size(), rd, error));

    // JoinAccept and Resync (one layout), with and without the generation info
    for (bool gen : {false, true})
    {
        JoinAccept a;
        a.epoch         = 7;
        a.hasGeneration = gen;
        if (gen) a.generation = generationInfoOf(w);
        bytes.clear();
        encodeJoinAccept(a, bytes);
        JoinAccept ad;
        ASSERT_TRUE(decodeJoinAccept(bytes.data(), bytes.size(), ad, error)) << error;
        EXPECT_EQ(ad.epoch, 7u);
        EXPECT_EQ(ad.hasGeneration, gen);
        EXPECT_TRUE(ad.generation == a.generation);
        for (size_t cut = 0; cut < bytes.size(); ++cut)
            EXPECT_FALSE(decodeJoinAccept(bytes.data(), cut, ad, error, "Resync")) << cut;
        std::vector<uint8_t> f = bytes;
        f[4] |= 0x10;
        EXPECT_FALSE(decodeJoinAccept(f.data(), f.size(), ad, error));
        f = bytes;
        f.push_back(0);
        EXPECT_FALSE(decodeJoinAccept(f.data(), f.size(), ad, error));
    }
}

TEST(ChunkMessages, RoundTripsAndBounds)
{
    std::string error;
    // ChunkInterest: entries, all, the entry bounds
    ChunkInterestMessage i;
    i.epoch = 3;
    i.space = 2;
    i.removes.push_back({1, 2});
    i.adds.push_back(ChunkInterestMessage::Add{3, 4, 99});
    std::vector<uint8_t> bytes;
    encodeChunkInterest(i, bytes);
    ChunkInterestMessage id;
    ASSERT_TRUE(decodeChunkInterest(bytes.data(), bytes.size(), id, error)) << error;
    EXPECT_EQ(id.epoch, 3u);
    EXPECT_EQ(id.space, 2);
    EXPECT_EQ(id.removes, i.removes);
    ASSERT_EQ(id.adds.size(), 1u);
    EXPECT_EQ(id.adds[0].ticket, 99u);
    ChunkInterestMessage empty;
    bytes.clear();
    encodeChunkInterest(empty, bytes);
    EXPECT_FALSE(decodeChunkInterest(bytes.data(), bytes.size(), id, error));  // 0 entries
    ChunkInterestMessage many;
    for (int k = 0; k < 4097; ++k) many.adds.push_back(ChunkInterestMessage::Add{k, 0, 1});
    bytes.clear();
    encodeChunkInterest(many, bytes);
    EXPECT_FALSE(decodeChunkInterest(bytes.data(), bytes.size(), id, error));
    ChunkInterestMessage all;
    all.all = true;
    bytes.clear();
    encodeChunkInterest(all, bytes);
    ASSERT_TRUE(decodeChunkInterest(bytes.data(), bytes.size(), id, error)) << error;
    EXPECT_TRUE(id.all);
    bytes.push_back(0);  // `all` with entries
    EXPECT_FALSE(decodeChunkInterest(bytes.data(), bytes.size(), id, error));

    // ChunkIndex: empty, 1000 entries; unsorted, duplicate, more than 2^18
    for (int n : {0, 1000})
    {
        ChunkIndexMessage x;
        x.epoch = 1;
        x.space = 0;
        for (int k = 0; k < n; ++k) x.chunks.push_back({k / 100, k % 100});
        bytes.clear();
        encodeChunkIndex(x, bytes);
        ChunkIndexMessage xd;
        ASSERT_TRUE(decodeChunkIndex(bytes.data(), bytes.size(), xd, error)) << error;
        EXPECT_EQ(xd.chunks, x.chunks);
    }
    for (auto chunks : {std::vector<std::pair<int32_t, int32_t>>{{1, 1}, {0, 0}},
                        std::vector<std::pair<int32_t, int32_t>>{{1, 1}, {1, 1}}})
    {
        ChunkIndexMessage x;
        x.chunks = chunks;
        bytes.clear();
        encodeChunkIndex(x, bytes);
        ChunkIndexMessage xd;
        EXPECT_FALSE(decodeChunkIndex(bytes.data(), bytes.size(), xd, error));
    }
    {
        std::vector<uint8_t> b{0, 0, 0, 0, 0, 0};
        writeVarint(b, kMaxIndexChunks + 1);
        ChunkIndexMessage xd;
        EXPECT_FALSE(decodeChunkIndex(b.data(), b.size(), xd, error));
    }

    // ChunkSnapshot: empty, record only, deltas only, both; raw and zstd
    auto roundTrip = [&](const ChunkSnapshotMessage& s)
    {
        std::vector<uint8_t> b;
        encodeChunkSnapshot(s, b);
        ChunkSnapshotMessage d;
        EXPECT_TRUE(decodeChunkSnapshot(b.data(), b.size(), d, error)) << error;
        EXPECT_EQ(d.epoch, s.epoch);
        EXPECT_EQ(d.ticket, s.ticket);
        EXPECT_EQ(d.chunk, s.chunk);
        EXPECT_EQ(d.sequence, s.sequence);
        EXPECT_EQ(d.N, s.N);
        EXPECT_EQ(d.hasRecord, s.hasRecord);
        if (s.hasRecord)
        {
            EXPECT_EQ(d.placementVersion, s.placementVersion);
            EXPECT_EQ(d.keys, s.keys);
        }
        EXPECT_EQ(d.hasDeltas, s.hasDeltas);
        if (s.hasDeltas)
        {
            EXPECT_EQ(d.deltas, s.deltas);
        }
        return b;
    };
    ChunkSnapshotMessage s;
    s.epoch    = 2;
    s.ticket   = 5;
    s.chunk    = chunkAt(1, 3, 4);
    s.sequence = 0;
    s.N        = 129;
    EXPECT_EQ(roundTrip(s).size(), 25u);  // an empty snapshot: the header only
    s.hasRecord        = true;
    s.placementVersion = 1;
    s.keys             = {InstanceKey{0, 3}, InstanceKey{0, 9}, InstanceKey{1, 2}, InstanceKey{255, (1u << 24) - 1}};
    roundTrip(s);
    s.hasDeltas = true;
    s.deltas.assign(129 * 129, 0);
    s.deltas[500]                    = kMaxTerrainDelta;
    const std::vector<uint8_t> small = roundTrip(s);  // sparse: zstd
    EXPECT_EQ(small[0] & kSnapshotZstd, kSnapshotZstd);
    for (size_t k = 0; k < s.deltas.size(); ++k) s.deltas[k] = int32_t((k * 2654435761u) % 20000) - 10000;
    roundTrip(s);
    s.hasRecord = false;
    roundTrip(s);
    // Errors: keys not ascending, a counter >= 2^24 (by hand), N > 257, an unknown flag bit, a delta above the bound,
    // bytes left over, more than 2^20 keys
    auto snapshotHeader = [](uint8_t flags, uint16_t N)
    {
        // flags, epoch, ticket, space, x, z, sequence (22 bytes of zeros), N
        std::vector<uint8_t> b(23, 0);
        b[0] = flags;
        b.push_back(uint8_t(N));
        b.push_back(uint8_t(N >> 8));
        return b;
    };
    ChunkSnapshotMessage sd;
    {
        std::vector<uint8_t> b = snapshotHeader(kSnapshotRecord, 129);
        b.push_back(1);
        b.push_back(0);
        writeVarint(b, 2);
        writeVarint(b, 5);
        writeVarint(b, 0);  // a repeated key
        EXPECT_FALSE(decodeChunkSnapshot(b.data(), b.size(), sd, error));
    }
    {
        std::vector<uint8_t> b = snapshotHeader(kSnapshotRecord, 129);
        b.push_back(1);
        b.push_back(0);
        writeVarint(b, 1);
        writeVarint(b, uint64_t(256) << 24);  // slot 256
        EXPECT_FALSE(decodeChunkSnapshot(b.data(), b.size(), sd, error));
    }
    {
        std::vector<uint8_t> b = snapshotHeader(kSnapshotRecord, 129);
        b.push_back(1);
        b.push_back(0);
        writeVarint(b, kMaxChunkTombstones + 1);
        for (size_t k = 0; k <= kMaxChunkTombstones; ++k) writeVarint(b, 1);
        EXPECT_FALSE(decodeChunkSnapshot(b.data(), b.size(), sd, error));
    }
    {
        const std::vector<uint8_t> b = snapshotHeader(0, 258);
        EXPECT_FALSE(decodeChunkSnapshot(b.data(), b.size(), sd, error));
        const std::vector<uint8_t> f = snapshotHeader(8, 129);
        EXPECT_FALSE(decodeChunkSnapshot(f.data(), f.size(), sd, error));
        std::vector<uint8_t> o = snapshotHeader(0, 129);
        o.push_back(0);
        EXPECT_FALSE(decodeChunkSnapshot(o.data(), o.size(), sd, error));
    }
    {
        std::vector<uint8_t> b = snapshotHeader(kSnapshotDeltas, 129);
        std::vector<uint8_t> v;
        writeVarint(v, zigzag(int64_t(kMaxTerrainDelta) + 1));
        for (int k = 1; k < 129 * 129; ++k) writeVarint(v, 0);
        writeVarint(b, v.size());
        b.insert(b.end(), v.begin(), v.end());
        EXPECT_FALSE(decodeChunkSnapshot(b.data(), b.size(), sd, error));
    }
}

// ---- the handshake --------------------------------------------------------------------------------------------------

namespace
{
struct ExtraComp
{
    static constexpr const char* name = "ExtraComp";
    static constexpr ComponentId id   = ComponentId(StreamingTest::StreamingTestComponent::Count);
    static constexpr uint32_t version = 1;
    int32_t v                         = 0;
    template <class V>
    void visit(V& vis)
    {
        vis("v", v);
    }
};
void extraComponents(Registry& r)
{
    StreamingTest::registerStreamingTestComponents(r);
    r.registerComponent<ExtraComp>(Replicate::Host);
}
void extraAssets(World& w)
{
    ContentTest::contentTestAssets(w);
    static const auto extra = ContentTest::contentTestBox(1, 1, 1, 0xFF00FF00u);
    w.registerMesh("test/extra", extra, w.findMaterial("test/content_flat"));
}
void extraGenerators(GeneratorList& l)
{
    ContentTest::registerContentTestGenerators<2, ContentTest::Variant::Committed>(l);
    l.add(9, std::make_shared<HookGenerator>("extra_hook", [](uint64_t, int64_t, int64_t) { return int32_t(0); }));
}
}  // namespace

TEST(Handshake, JoinOrderAndAdmission)
{
    LoopbackOptions o;
    o.delayRounds = 2;
    NetPeers n(worldNetTestSetup(false), 1, o);
    World& h = *n.host;
    World& c = n.client();
    // A host entity of the spawn set
    const Entity before = h.create();
    h.step();
    c.step();
    // The client's first message is its JoinRequest; nothing else leaves it before Joined
    EXPECT_EQ(c.netStats().messagesOut, 1u);
    EXPECT_EQ(c.joinState(), JoinState::Joining);
    // A message the client sends before Joined is dropped (counted)
    ChunkInterestMessage all;
    all.all = true;
    std::vector<uint8_t> bytes;
    encodeChunkInterest(all, bytes);
    c.sendRaw(0, MessageId(EngineMessage::ChunkInterest), bytes.data(), bytes.size());
    EXPECT_EQ(c.netStats().droppedNotJoined, 1u);
    // Admission: the host's lists contain the peer from the step that admitted it
    int admitted = -1;
    for (int k = 0; k < 16 && admitted < 0; ++k)
    {
        h.step();
        if (!h.connectedPeers().empty())
        {
            admitted = k;
            EXPECT_EQ(h.connectedPeers(), std::vector<uint8_t>{1});
            EXPECT_EQ(h.peersJoined(), std::vector<uint8_t>{1});
        }
        else
            EXPECT_TRUE(h.peersJoined().empty());
        c.step();
        // JoinAccept before any spawn: a replicated entity implies Joined
        if (!c.replication().netIds().empty())
        {
            EXPECT_EQ(c.joinState(), JoinState::Joined);
        }
    }
    ASSERT_GE(admitted, 0);
    EXPECT_TRUE(n.stepUntil([&] { return c.joinState() == JoinState::Joined; }, 16));
    EXPECT_EQ(c.connectedPeers(), std::vector<uint8_t>{0});
    EXPECT_TRUE(n.stepUntil([&] { return c.replication().entityOfNetId(h.replication().netIdOf(before)).valid(); }));
    EXPECT_EQ(c.joinRejection(), "");
    // A second JoinRequest of an admitted peer is ignored (rejected input)
    const uint64_t rejected = h.netStats().rejected;
    c.replication().sendJoinRequest();
    n.step(4);
    EXPECT_EQ(h.netStats().rejected, rejected + 1);
    EXPECT_EQ(h.connectedPeers(), std::vector<uint8_t>{1});
    EXPECT_EQ(h.netStats().joinsRejected, 0u);
}

namespace
{
Entity g_policyPlayer;
// A join policy in phase 4 of the admission step: the new peer's player (in the spawn set of that step's phase 8)
void joinPolicySystem(World& w, double)
{
    for (uint8_t p : w.peersJoined())
    {
        const Entity e = w.create();
        Player pl;
        pl.peer = p;
        w.add<Player>(e, pl);
        w.setOwner(e, p);
        g_policyPlayer = e;
    }
}
void joinPolicySystems(SystemList& l)
{
    l.add(Phase::Gameplay, "join_policy", joinPolicySystem, RunOn::Host);
}
}  // namespace

TEST(Handshake, ControlBeforeAcceptIsDeferred)
{
    WorldSetup s;
    s.profiling = false;
    s.systems   = joinPolicySystems;
    LoopbackOptions o;
    // The State packet of JoinAccept and the spawn set takes several rounds, the OwnerCommand (Control) one
    o.bytesPerRound = 48;
    g_policyPlayer  = Entity();
    NetPeers n(s, 1, o);
    World& h = *n.host;
    World& c = n.client();
    // Right after the admission step (its listener): the policy teleports the new player, an OwnerCommand on Control
    // visible to the client in the same round as the State packet of JoinAccept and the spawn set
    const Vec3 target(40, 0, 7);
    bool teleported = false;
    h.addStepListener(
        [&]()
        {
            if (!teleported && g_policyPlayer.valid())
            {
                teleported = true;
                h.ownerTeleport(g_policyPlayer, target, Quat::Identity());
            }
            return true;
        });
    bool controlWhileJoining = false;
    auto* ct                 = static_cast<LoopbackTransport*>(c.transport());
    for (int k = 0; k < 64 && c.joinState() != JoinState::Joined; ++k)
    {
        const uint64_t control = ct->stats().packetsIn[size_t(Channel::Control)];
        c.step();
        if (c.joinState() == JoinState::Joining && ct->stats().packetsIn[size_t(Channel::Control)] > control)
            controlWhileJoining = true;
        h.step();
    }
    ASSERT_EQ(c.joinState(), JoinState::Joined);
    EXPECT_TRUE(controlWhileJoining) << "the OwnerCommand overtook JoinAccept";
    ASSERT_TRUE(
        n.stepUntil([&] { return c.replication().entityOfNetId(h.replication().netIdOf(g_policyPlayer)).valid(); }));
    const Entity mine = c.replication().entityOfNetId(h.replication().netIdOf(g_policyPlayer));
    EXPECT_TRUE((c.read<Transform>(mine).position() - target).norm() < 1e-9) << c.read<Transform>(mine).position();
    EXPECT_EQ(c.netStats().rejected, 0u);
}

// The acceptance test of S26: a join with another schema hash is rejected with a message on both sides
TEST(Handshake, SchemaHashMismatchIsRejected)
{
    NetPeers n(worldNetTestSetup(false), 1, {},
               [](WorldSetup& s, int peer)
               {
                   if (peer == 1) s.components = extraComponents;
               });
    World& h = *n.host;
    World& c = n.client();
    h.create();
    n.step(6);
    EXPECT_EQ(h.netStats().joinsRejected, 1u);
    EXPECT_TRUE(h.connectedPeers().empty());
    EXPECT_EQ(c.joinState(), JoinState::Rejected);
    EXPECT_NE(c.joinRejection().find("schema"), std::string::npos) << c.joinRejection();
    EXPECT_NE(c.joinRejection().find("types"), std::string::npos) << c.joinRejection();
    EXPECT_TRUE(c.replication().netIds().empty());
    EXPECT_TRUE(c.connectedPeers().empty());
    // Nothing was sent but the JoinReject
    auto* ct = static_cast<LoopbackTransport*>(c.transport());
    EXPECT_EQ(ct->stats().packetsIn[size_t(Channel::Control)], 1u);
    EXPECT_EQ(ct->stats().packetsIn[size_t(Channel::State)], 0u);
    EXPECT_EQ(ct->stats().packetsIn[size_t(Channel::ChunkData)], 0u);
}

TEST(Handshake, RejectionReasons)
{
    struct Case
    {
        const char* text;
        void (*adjust)(WorldSetup&);
    };
    const Case cases[] = {
        {"environment: the host's world has WorldTime and Weather disabled",
         [](WorldSetup& s) { s.environment.enabled = true; }},
        {"render name tables differ", [](WorldSetup& s) { s.renderAssets = extraAssets; }},
        {"seed 7, the host's", [](WorldSetup& s) { s.seed = 7; }},
        {"generators: the host's [flat, test_dungeon, test_terrain], this client's [extra_hook, flat, test_dungeon, "
         "test_terrain]",
         [](WorldSetup& s) { s.generators = extraGenerators; }},
        {"test_terrain v1 on this client, the host places v2",
         [](WorldSetup& s)
         {
             // Another generator build: its own content cache
             s.generators    = ContentTest::registerContentTestGenerators<1, ContentTest::Variant::Committed>;
             s.sharedContent = std::make_shared<ContentCache>();
         }},
    };
    for (const Case& k : cases)
    {
        NetPeers n(worldNetTestSetup(false), 1, {},
                   [&](WorldSetup& s, int peer)
                   {
                       if (peer == 1) k.adjust(s);
                   });
        n.step(4);
        EXPECT_EQ(n.client().joinState(), JoinState::Rejected) << k.text;
        EXPECT_NE(n.client().joinRejection().find(k.text), std::string::npos) << n.client().joinRejection();
        EXPECT_EQ(n.host->netStats().joinsRejected, 1u) << k.text;
    }
}

TEST(Handshake, ProtocolVersionMismatchIsRejected)
{
    auto net      = LoopbackTransport::createNetwork(2);
    WorldSetup hs = worldNetTestSetup(false);
    hs.transport  = net[0].get();
    World host(hs);
    // Version 2 with a layout of another shape after the version
    const std::vector<uint8_t> request{2, 0, 0, 0, 0xAB, 0xCD, 0xEF};
    net[1]->send(0, Channel::Control, framed(MessageId(EngineMessage::JoinRequest), request), Reliability::Reliable);
    host.step();
    host.step();
    const auto got = receivedMessages(*net[1]);
    ASSERT_EQ(got.size(), 1u);
    EXPECT_EQ(got[0].first, MessageId(EngineMessage::JoinReject));
    // Decodable with the frozen prefix: the reason first
    ASSERT_GE(got[0].second.size(), 1u);
    EXPECT_EQ(got[0].second[0], uint8_t(JoinRejectReason::ProtocolVersion));
    JoinReject r;
    std::string error;
    ASSERT_TRUE(decodeJoinReject(got[0].second.data(), got[0].second.size(), r, error)) << error;
    EXPECT_EQ(r.reason, JoinRejectReason::ProtocolVersion);
    EXPECT_TRUE(host.connectedPeers().empty());
}

TEST(Handshake, MalformedRequestAndEarlyMessages)
{
    auto net      = LoopbackTransport::createNetwork(2);
    WorldSetup hs = worldNetTestSetup(false);
    hs.transport  = net[0].get();
    World host(hs);
    host.step();
    // A ChunkInterest and a PlayerAction before admission: rejected input, no state change
    ChunkInterestMessage all;
    all.all = true;
    std::vector<uint8_t> interest;
    encodeChunkInterest(all, interest);
    net[1]->send(0, Channel::Control, framed(MessageId(EngineMessage::ChunkInterest), interest), Reliability::Reliable);
    std::vector<uint8_t> action;
    toBytes(PlayerActionMessage{}, action, nullptr);
    net[1]->send(0, Channel::Control, framed(MessageId(EngineMessage::PlayerAction), action), Reliability::Reliable);
    const uint64_t rejected = host.netStats().rejected;
    host.step();
    EXPECT_EQ(host.netStats().rejected, rejected + 2);
    EXPECT_TRUE(host.replication().subscriptions().peers().empty());
    // A garbage JoinRequest of this version: JoinReject{Malformed}, counted
    const std::vector<uint8_t> garbage{1, 0, 0, 0, 1, 2, 3};
    net[1]->send(0, Channel::Control, framed(MessageId(EngineMessage::JoinRequest), garbage), Reliability::Reliable);
    host.step();
    host.step();
    EXPECT_EQ(host.netStats().rejected, rejected + 3);
    EXPECT_EQ(host.netStats().joinsRejected, 1u);
    bool malformed = false;
    for (const auto& m : receivedMessages(*net[1]))
        malformed = malformed || (m.first == MessageId(EngineMessage::JoinReject) && !m.second.empty() &&
                                  m.second[0] == uint8_t(JoinRejectReason::Malformed));
    EXPECT_TRUE(malformed);
    EXPECT_TRUE(host.connectedPeers().empty());
}

namespace
{
// A v1 world's save with a tombstone in kC (its record at placement version 1)
std::vector<uint8_t> v1SaveWithTombstone(InstanceKey& key)
{
    // Another generator build: its own content cache (one per process, shared by the tests that load it)
    static std::shared_ptr<ContentCache> cache = std::make_shared<ContentCache>();
    World v1(worldNetTestSetup(false, 1, cache));
    key = acceptedKey(v1, kC);
    v1.tombstone(kC, key);
    return v1.save();
}
}  // namespace

// E11: a fresh v2 client adopts the pins of a host that loaded a v1 save
TEST(Handshake, ClientAdoptsOlderPins)
{
    InstanceKey key;
    const std::vector<uint8_t> blob = v1SaveWithTombstone(key);
    WorldSetup hs                   = worldNetTestSetup(false, 2);
    NetPeers n(hs, 1, {}, {});
    // The host loads before the client joins (the handshake starts at the first steps)
    ASSERT_TRUE(n.host->tryLoad(blob).ok());
    ASSERT_EQ(n.host->generation().space(0).pinned, 1u);
    ASSERT_TRUE(n.stepUntil([&] { return synced(n); }));
    World& c = n.client();
    EXPECT_EQ(c.joinState(), JoinState::Joined);
    EXPECT_EQ(c.generation().space(0).pinned, 1u);
    EXPECT_TRUE(c.generationChecked());
    Rng rng(5);
    for (int k = 0; k < 64; ++k)
    {
        const double x = double(rng.range(0, 4095)), z = double(rng.range(0, 4095));
        EXPECT_EQ(c.terrainHeight(0, x, z), n.host->terrainHeight(0, x, z)) << x << ", " << z;
    }
    // The chunk's snapshot creates the v1 record with its tombstone
    ASSERT_NE(c.chunkRecord(kC), nullptr);
    EXPECT_EQ(c.chunkRecord(kC)->placementVersion, 1u);
    EXPECT_TRUE(c.isTombstoned(kC, key));
    expectChunkEqual(*n.host, c, kC);
}

// A host load with other pins: Resync v2, the client adopts them, clears its chunk state, subscribes again in the new
// epoch and ends equal; an interest of the old epoch in flight is dropped
TEST(Handshake, HostLoadResyncAdoptsNewPins)
{
    InstanceKey key;
    const std::vector<uint8_t> blob = v1SaveWithTombstone(key);
    NetPeers n(worldNetTestSetup(false, 2));
    ASSERT_TRUE(n.stepUntil([&] { return synced(n); }));
    World& c = n.client();
    n.host->editTerrain(0, raiseBrush(1500, 900));
    ASSERT_TRUE(n.stepUntil([&] { return synced(n) && c.terrainEdits().chunkCount() > 0; }));
    ASSERT_EQ(c.generation().space(0).pinned, 2u);
    ASSERT_TRUE(n.host->tryLoad(blob).ok());
    // An interest of the old epoch (sent by the client before it saw the Resync)
    ChunkInterestMessage old;
    old.all   = true;
    old.epoch = 0;
    std::vector<uint8_t> bytes;
    encodeChunkInterest(old, bytes);
    c.sendRaw(0, MessageId(EngineMessage::ChunkInterest), bytes.data(), bytes.size());
    const uint64_t droppedEpoch = n.host->replication().subscriptions().stats().droppedEpoch;
    n.step();
    EXPECT_EQ(c.replication().epoch(), 1u);
    EXPECT_EQ(c.generation().space(0).pinned, 1u);
    ASSERT_TRUE(n.stepUntil([&] { return synced(n); }));
    EXPECT_EQ(n.host->replication().subscriptions().stats().droppedEpoch, droppedEpoch + 1);
    EXPECT_EQ(c.terrainEdits().chunkCount(), n.host->terrainEdits().chunkCount());
    EXPECT_TRUE(c.isTombstoned(kC, key));
    expectChunkEqual(*n.host, c, kC);
    EXPECT_EQ(c.netStats().modificationGaps, 0u);
    EXPECT_EQ(c.netStats().rejected, 0u);
}

// ---- chunk sync -----------------------------------------------------------------------------------------------------

TEST(ChunkSync, SubscribeSnapshotAndLiveEdits)
{
    NetPeers n(worldNetTestSetup(true));
    World& c = n.client();
    ASSERT_TRUE(n.stepUntil([&] { return c.joinState() == JoinState::Joined; }));
    c.replication().chunkSync().hold(kC);
    ASSERT_TRUE(n.stepUntil([&] { return c.chunkSynced(kC); }));
    EXPECT_EQ(c.replication().chunkSync().stats().snapshotsApplied, 1u);
    n.host->editTerrain(0, raiseBrush(centreOf(kC).x(), centreOf(kC).z()));
    const InstanceKey key = acceptedKey(*n.host, kC);
    n.host->tombstone(kC, key);
    n.step(2);
    // The edit's part and its tombstones-on-edit, then the tombstone: every sequence applied live
    EXPECT_GE(c.terrainEdits().sequence(kC), 2u);
    EXPECT_TRUE(c.isTombstoned(kC, key));
    expectChunkEqual(*n.host, c, kC);
    EXPECT_EQ(c.replication().chunkSync().stats().snapshotsApplied, 1u);
    EXPECT_EQ(c.netStats().modificationGaps, 0u);
}

TEST(ChunkSync, LiveEditsOnlyToSubscribers)
{
    NetPeers n(worldNetTestSetup(true), 2);
    ASSERT_TRUE(n.stepUntil([&] { return n.allJoined(); }));
    n.client(1).replication().chunkSync().hold(kC);
    n.client(2).replication().chunkSync().hold(kD);
    ASSERT_TRUE(n.stepUntil([&] { return n.client(1).chunkSynced(kC) && n.client(2).chunkSynced(kD); }));
    const uint64_t filtered = n.host->replication().subscriptions().stats().partsFiltered;
    // Across the border of C and D
    n.host->editTerrain(0, raiseBrush(double(kD.x) * 256, centreOf(kC).z()));
    ASSERT_GT(n.host->terrainEdits().sequence(kC), 0u);
    ASSERT_GT(n.host->terrainEdits().sequence(kD), 0u);
    n.step(2);
    // Each client got its chunk's part only (one part filtered per client)
    EXPECT_EQ(n.host->replication().subscriptions().stats().partsFiltered - filtered, 2u);
    EXPECT_EQ(n.client(1).terrainEdits().entry(kD), nullptr);
    EXPECT_EQ(n.client(2).terrainEdits().entry(kC), nullptr);
    expectChunkEqual(*n.host, n.client(1), kC);
    expectChunkEqual(*n.host, n.client(2), kD);
}

// The acceptance test of S26: a client that reloads a chunk while an edit is in flight ends with the host's heights;
// over the release offsets k = -1..3 relative to the edit, one or two edits, delay 2
TEST(ChunkSync, ReloadDuringEditConverges)
{
    uint64_t queued = 0, dropped = 0, stale = 0;
    for (int edits = 1; edits <= 2; ++edits)
        for (int k = -1; k <= 3; ++k)
        {
            LoopbackOptions o;
            o.delayRounds = 2;
            NetPeers n(worldNetTestSetup(true), 1, o);
            World& c           = n.client();
            ClientChunkSync& s = c.replication().chunkSync();
            ASSERT_TRUE(n.stepUntil([&] { return c.joinState() == JoinState::Joined; }));
            s.hold(kC);
            s.hold(kD);
            ASSERT_TRUE(n.stepUntil([&] { return c.chunkSynced(kC) && c.chunkSynced(kD); }));
            for (int round = -2; round <= 4; ++round)
            {
                if (round == k) s.release(kC);
                if (round == k + 1) s.hold(kC);
                if (round == 0 || (edits == 2 && round == 1))
                    n.host->editTerrain(0, raiseBrush(double(kD.x) * 256, centreOf(kC).z() + 10 * round, 24, 2));
                n.step();
            }
            ASSERT_TRUE(n.stepUntil([&] { return synced(n); })) << "k " << k << ", edits " << edits;
            expectChunkEqual(*n.host, c, kC);
            expectChunkEqual(*n.host, c, kD);
            for (int64_t z = 5 * 128; z <= 6 * 128; z += 7)
                for (int64_t x = 4 * 128; x <= 6 * 128; x += 7)
                    ASSERT_EQ(c.terrainSample(0, x, z), n.host->terrainSample(0, x, z)) << x << ", " << z;
            EXPECT_EQ(c.netStats().modificationGaps, 0u);
            queued += s.stats().partsQueued;
            dropped += s.stats().partsDroppedOld + s.stats().partsDroppedNotSubscribed;
            stale += s.stats().snapshotsStale;
        }
    // Across the parameter set: live parts arrived while C was pending (queued, then dropped or applied after the
    // snapshot), and a snapshot of an earlier subscription was stale
    EXPECT_GT(queued, 0u);
    EXPECT_GT(dropped, 0u);
    std::cout << "parts queued " << queued << ", dropped " << dropped << ", stale snapshots " << stale << std::endl;
}

TEST(ChunkSync, UnloadDropsStateKeepsSharedBorder)
{
    NetPeers n(worldNetTestSetup(true));
    World& c           = n.client();
    ClientChunkSync& s = c.replication().chunkSync();
    ASSERT_TRUE(n.stepUntil([&] { return c.joinState() == JoinState::Joined; }));
    s.hold(kC);
    s.hold(kD);
    ASSERT_TRUE(n.stepUntil([&] { return c.chunkSynced(kC) && c.chunkSynced(kD); }));
    // An edit across the border, one inside C, a tombstone in C
    n.host->editTerrain(0, raiseBrush(double(kD.x) * 256, centreOf(kC).z()));
    n.host->editTerrain(0, raiseBrush(centreOf(kC).x() - 40, centreOf(kC).z()));
    n.host->tombstone(kC, acceptedKey(*n.host, kC));
    n.step(2);
    expectChunkEqual(*n.host, c, kC);
    int recordChanges = 0;
    c.addRecordListener([&](const ChunkId& x) { recordChanges += x == kC ? 1 : 0; });
    s.release(kC);
    n.step();
    EXPECT_EQ(s.state(kC), ClientChunkSync::State::None);
    EXPECT_EQ(c.terrainEdits().entry(kC), nullptr);
    EXPECT_EQ(c.chunkRecord(kC), nullptr);
    EXPECT_EQ(recordChanges, 1);
    // The interior reads 0, the shared border with D keeps the host's values
    const int64_t S = 128;
    EXPECT_EQ(c.terrainEdits().delta(0, kC.x * S + 40, kC.z * S + 64), 0);
    bool borderEdited = false;
    for (int64_t k = kC.z * S; k <= kC.z * S + S; ++k)
    {
        const int64_t i = kD.x * S;
        EXPECT_EQ(c.terrainEdits().delta(0, i, k), n.host->terrainEdits().delta(0, i, k));
        borderEdited = borderEdited || n.host->terrainEdits().delta(0, i, k) != 0;
    }
    EXPECT_TRUE(borderEdited);
    expectChunkEqual(*n.host, c, kD);
    // One terrain change of C's box (the drop): a TerrainEditedEvent of the commit, readable in the next step
    EXPECT_EQ(c.events.find(TerrainEditedEvent::id)->pendingCount(), 1u);
}

TEST(ChunkSync, GapRerequests)
{
    NetPeers n(worldNetTestSetup(true));
    World& c           = n.client();
    ClientChunkSync& s = c.replication().chunkSync();
    ASSERT_TRUE(n.stepUntil([&] { return c.joinState() == JoinState::Joined; }));
    s.hold(kC);
    ASSERT_TRUE(n.stepUntil([&] { return c.chunkSynced(kC); }));
    n.host->editTerrain(0, raiseBrush(centreOf(kC).x(), centreOf(kC).z()));
    ASSERT_TRUE(n.stepUntil([&] { return c.terrainEdits().sequence(kC) == n.host->terrainEdits().sequence(kC); }));
    for (bool tombstone : {false, true})
    {
        const uint32_t current = c.terrainEdits().sequence(kC);
        const uint32_t ticket  = s.ticket(kC);
        std::vector<uint8_t> m;
        if (tombstone)
        {
            TombstoneEntry e;
            e.x                = kC.x;
            e.z                = kC.z;
            e.placementVersion = c.effectivePlacementVersion(kC);
            e.sequence         = current + 2;
            e.key              = acceptedKey(*n.host, kC);
            encodeTombstoneModification(c.replication().epoch(), 0, {e}, m);
            s.onTombstoneModification(0, m.data(), m.size());
        }
        else
        {
            m = modificationBytes(c.replication().epoch(), 0, {partOf(kC, current + 2, 10, 10, 2, 2, 777)});
            s.onTerrainModification(0, m.data(), m.size());
        }
        EXPECT_EQ(c.terrainEdits().sequence(kC), current);
        EXPECT_EQ(c.netStats().modificationGaps, tombstone ? 2u : 1u);
        EXPECT_EQ(c.netStats().chunkRerequests, tombstone ? 2u : 1u);
        EXPECT_EQ(s.state(kC), ClientChunkSync::State::Pending);
        EXPECT_NE(s.ticket(kC), ticket);
        // The next phase 8 sends an add with the new ticket; the client ends equal to the host
        const uint64_t interest = s.stats().interestMessages;
        n.step();
        EXPECT_EQ(s.stats().interestMessages, interest + 1);
        ASSERT_TRUE(n.stepUntil([&] { return c.chunkSynced(kC); }));
        expectChunkEqual(*n.host, c, kC);
        EXPECT_EQ(c.terrainEdits().delta(0, kC.x * 128 + 10, kC.z * 128 + 10),
                  n.host->terrainEdits().delta(0, kC.x * 128 + 10, kC.z * 128 + 10));
    }
}

TEST(ChunkSync, QueueOverflowRerequests)
{
    LoopbackOptions o;
    o.delayRounds = 2;
    NetPeers n(worldNetTestSetup(true), 1, o);
    World& c           = n.client();
    ClientChunkSync& s = c.replication().chunkSync();
    ASSERT_TRUE(n.stepUntil([&] { return c.joinState() == JoinState::Joined; }));
    s.hold(kC);
    n.step();
    ASSERT_EQ(s.state(kC), ClientChunkSync::State::Pending);
    const uint32_t ticket = s.ticket(kC);
    for (uint32_t q = 1; q <= ClientChunkSync::kMaxQueuedPartsPerChunk + 1; ++q)
    {
        const std::vector<uint8_t> m = modificationBytes(c.replication().epoch(), 0, {partOf(kC, q, 5, 5, 1, 1, 3)});
        s.onTerrainModification(0, m.data(), m.size());
    }
    EXPECT_EQ(s.stats().queueOverflows, 1u);
    EXPECT_EQ(s.queuedParts(kC), 0u);
    EXPECT_NE(s.ticket(kC), ticket);
    ASSERT_TRUE(n.stepUntil([&] { return c.chunkSynced(kC); }));
    EXPECT_EQ(s.stats().snapshotsStale, 1u);
    expectChunkEqual(*n.host, c, kC);
    EXPECT_EQ(c.terrainEdits().entry(kC), nullptr);  // the host never edited C
}

TEST(ChunkSync, InvalidSnapshotsApplyNothing)
{
    NetPeers n(worldNetTestSetup(true));
    World& c           = n.client();
    ClientChunkSync& s = c.replication().chunkSync();
    ASSERT_TRUE(n.stepUntil([&] { return c.joinState() == JoinState::Joined; }));
    // A pending chunk whose snapshot is held back by the host's budget is not needed: the crafted ones go straight to
    // the handler
    const SpaceId dungeon = StreamingTest::spaceNamed(c, "test_dungeon");
    const SpaceId flat    = StreamingTest::spaceNamed(c, "test_flat");
    auto base             = [&](const ChunkId& chunk)
    {
        ChunkSnapshotMessage m;
        m.epoch  = c.replication().epoch();
        m.ticket = 1;
        m.chunk  = chunk;
        m.N      = c.spaceTemplate(chunk.space).kind == SpaceKind::OpenWorld ? 129 : 0;
        return m;
    };
    std::vector<std::pair<std::string, ChunkSnapshotMessage>> bad;
    {
        auto m        = base(kC);
        m.chunk.space = 99;
        bad.push_back({"an unknown space", m});
    }
    bad.push_back({"a chunk out of range", base(chunkAt(0, 16, 0))});
    {
        auto m = base(kC);
        m.N    = 257;
        bad.push_back({"another N", m});
    }
    {
        auto m      = base(chunkAt(dungeon, 1, 1));
        m.N         = 129;
        m.hasDeltas = true;
        m.deltas.assign(129 * 129, 1);
        bad.push_back({"deltas for an enclosed space", m});
    }
    {
        auto m             = base(kC);
        m.hasRecord        = true;
        m.placementVersion = 3;
        bad.push_back({"a placement version above the current one", m});
    }
    {
        auto m             = base(chunkAt(flat, 1, 1));
        m.hasRecord        = true;
        m.placementVersion = 1;
        m.keys             = {InstanceKey{0, 1}};
        bad.push_back({"keys for a slotless generator", m});
    }
    {
        auto m             = base(kC);
        m.hasRecord        = true;
        m.placementVersion = 2;
        m.keys             = {InstanceKey{5, 1}};
        bad.push_back({"a slot out of range", m});
    }
    {
        auto m             = base(kC);
        m.hasRecord        = true;
        m.placementVersion = 2;
        m.keys             = {InstanceKey{0, 1u << 23}};
        bad.push_back({"a counter out of range", m});
    }
    const uint64_t hash = c.hash();
    for (const auto& [what, m] : bad)
    {
        std::vector<uint8_t> bytes;
        encodeChunkSnapshot(m, bytes);
        const uint64_t rejected = c.netStats().rejected;
        s.onSnapshot(0, bytes.data(), bytes.size());
        EXPECT_EQ(c.netStats().rejected, rejected + 1) << what;
        EXPECT_EQ(c.hash(), hash) << what;
    }
    EXPECT_EQ(s.stats().snapshotsApplied, 0u);
    // Another epoch: dropped, not counted; an unknown ticket: stale
    auto other  = base(kC);
    other.epoch = 9;
    std::vector<uint8_t> bytes;
    encodeChunkSnapshot(other, bytes);
    const uint64_t rejected = c.netStats().rejected;
    s.onSnapshot(0, bytes.data(), bytes.size());
    EXPECT_EQ(c.netStats().rejected, rejected);
    EXPECT_EQ(s.stats().snapshotsStale, 0u);
    bytes.clear();
    encodeChunkSnapshot(base(kC), bytes);
    s.onSnapshot(0, bytes.data(), bytes.size());
    EXPECT_EQ(s.stats().snapshotsStale, 1u);
    EXPECT_EQ(c.netStats().rejected, rejected);
}

TEST(ChunkSync, HostValidatesInterest)
{
    WorldSetup setup                      = worldNetTestSetup(true);
    setup.config.network.maxChunksPerPeer = 4;
    NetPeers n(setup);
    World& h = *n.host;
    World& c = n.client();
    ASSERT_TRUE(n.stepUntil([&] { return c.joinState() == JoinState::Joined; }));
    HostChunkSubscriptions& subs = h.replication().subscriptions();
    auto send                    = [&](const ChunkInterestMessage& m, uint8_t from = 1)
    {
        std::vector<uint8_t> b;
        encodeChunkInterest(m, b);
        subs.onInterest(from, b.data(), b.size());
    };
    ChunkInterestMessage m;
    m.epoch          = h.replication().epoch();
    const uint64_t r = h.netStats().rejected;
    m.adds           = {ChunkInterestMessage::Add{16, 0, 1}};
    send(m);  // out of range
    m.adds = {ChunkInterestMessage::Add{1, 1, 1}, ChunkInterestMessage::Add{1, 1, 2}};
    send(m);  // twice
    m.adds.clear();
    for (int k = 0; k < 5; ++k) m.adds.push_back(ChunkInterestMessage::Add{k, 0, 1});
    send(m);  // more than maxChunksPerPeer
    ChunkInterestMessage all;
    all.all   = true;
    all.epoch = m.epoch;
    send(all);  // `all` in a streaming world
    EXPECT_EQ(h.netStats().rejected, r + 4);
    EXPECT_EQ(subs.subscribedCount(1), 0u);
    EXPECT_EQ(subs.queuedSnapshots(1), 0u);
    // From a client role
    const uint64_t cr = c.netStats().rejected;
    std::vector<uint8_t> b;
    encodeChunkInterest(m, b);
    c.replication().subscriptions().onInterest(0, b.data(), b.size());
    EXPECT_EQ(c.netStats().rejected, cr + 1);
}

TEST(ChunkSync, SnapshotBudgetPerStep)
{
    WorldSetup setup                               = worldNetTestSetup(true);
    setup.config.network.chunkSnapshotBytesPerStep = 1;
    NetPeers n(setup);
    World& h           = *n.host;
    World& c           = n.client();
    ClientChunkSync& s = c.replication().chunkSync();
    ASSERT_TRUE(n.stepUntil([&] { return c.joinState() == JoinState::Joined; }));
    std::vector<ChunkId> chunks;
    for (int32_t x = 2; x < 10; ++x)
    {
        chunks.push_back(chunkAt(0, x, 10));
        h.editTerrain(0, raiseBrush(centreOf(chunks.back()).x(), centreOf(chunks.back()).z()));
    }
    for (const ChunkId& x : chunks) s.hold(x);
    c.step();  // the interest
    h.step();  // received: 8 queued, the first served
    EXPECT_EQ(h.replication().subscriptions().stats().snapshotsSent, 1u);
    EXPECT_EQ(h.replication().subscriptions().queuedSnapshots(1), 7u);
    // A resubscription of a queued chunk replaces its ticket: one snapshot
    s.release(chunks[5]);
    c.step();
    s.hold(chunks[5]);
    c.step();
    for (int k = 0; k < 3; ++k) n.step();
    EXPECT_EQ(h.replication().subscriptions().stats().snapshotsSent, 4u);
    ASSERT_TRUE(n.stepUntil([&] { return synced(n); }));
    EXPECT_EQ(h.replication().subscriptions().stats().snapshotsSent, 8u);
    for (const ChunkId& x : chunks) expectChunkEqual(h, c, x);
    EXPECT_EQ(s.stats().snapshotsStale, 0u);
}

// §17.10's join data: 50 chunks of a 1 m space edited with a seeded noise brush covering each chunk; about 0.3 s (the
// edits of 50 full 1 m chunks; the snapshots' encode and apply are what is measured)
TEST(ChunkSync, JoinDataFor50EditedChunks)
{
    NetPeers n(worldNetTestSetup(true));
    World& h             = *n.host;
    World& c             = n.client();
    const SpaceId valley = StreamingTest::spaceNamed(h, "test_valley");
    std::vector<ChunkId> chunks;
    // The seeded noise deltas of the 50 chunks written into the host's store (the state an edit leaves; brushes over
    // 50 full 1 m chunks cost seconds of base heights, which this test does not measure)
    TerrainEditStore& st = h.terrainEditsForUpdate();
    const int64_t S      = st.chunkSamples(valley);
    for (int64_t k = 0; k < 5 * S; ++k)
        for (int64_t i = 0; i < 10 * S; ++i)
        {
            const int32_t v =
                int32_t(hashCombine(uint64_t(i), uint64_t(k)) % 4000) - 2000 + int32_t((i * 7 + k * 3) % 512);
            st.reserveFor(valley, i, k, v);
            st.writeDelta(valley, i, k, v);
        }
    for (int32_t z = 0; z < 5; ++z)
        for (int32_t x = 0; x < 10; ++x)
        {
            chunks.push_back(chunkAt(valley, x, z));
            st.bumpSequence(chunks.back());
        }
    ASSERT_TRUE(n.stepUntil([&] { return c.joinState() == JoinState::Joined; }));
    const auto t0 = std::chrono::steady_clock::now();
    for (const ChunkId& x : chunks) c.replication().chunkSync().hold(x);
    n.step();
    ASSERT_TRUE(n.stepUntil([&] { return synced(n); }, 200));
    const double ms      = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    const uint64_t bytes = h.replication().subscriptions().stats().snapshotBytes;
    std::cout << "join data of 50 edited 1 m chunks: " << bytes << " bytes of snapshots, " << ms << " ms" << std::endl;
    EXPECT_LE(bytes, uint64_t(20) << 20);
    for (const ChunkId& x : chunks) expectChunkEqual(h, c, x);
}

// S24's gate and hold: a client-owned player on C whose snapshot is held back by the budget stays held
TEST(ChunkSync, ResidentAfterSnapshot)
{
    WorldSetup setup                               = worldNetTestSetup(true);
    setup.config.network.chunkSnapshotBytesPerStep = 1;
    LoopbackOptions o;
    o.delayRounds = 2;
    NetPeers n(setup, 1, o);
    World& h = *n.host;
    World& c = n.client();
    ASSERT_TRUE(n.stepUntil([&] { return c.joinState() == JoinState::Joined; }));
    // Many held chunks queued before the player's: its snapshot waits a few steps behind them
    for (int32_t x = 0; x < 6; ++x) c.replication().chunkSync().hold(chunkAt(0, x, 12));
    n.step();
    const Entity player                            = clientPlayer(h, centreOf(kC));
    h.get<CharacterIntent>(player).desiredVelocity = Vec3(1, 0, 0);
    Entity mine;
    ASSERT_TRUE(n.stepUntil(
        [&]
        {
            mine = c.replication().entityOfNetId(h.replication().netIdOf(player));
            return mine.valid();
        }));
    bool sawHeld = false;
    for (int k = 0; k < 64 && !c.chunkSynced(kC); ++k)
    {
        if (c.residency().state(kC) == ChunkState::Requested &&
            c.residency().partsReady(kC) == c.residency().partsWanted(kC))
        {
            sawHeld = true;
            EXPECT_FALSE(c.chunkReady(0, centreOf(kC).x(), centreOf(kC).z()));
        }
        n.step();
    }
    ASSERT_TRUE(c.chunkSynced(kC));
    EXPECT_TRUE(sawHeld);
    EXPECT_GT(c.residency().stats().moverHolds, 0u);
    EXPECT_GT(c.residency().stats().streamingStalls, 0u);
    n.step();
    EXPECT_EQ(c.residency().state(kC), ChunkState::Resident);
    const Vec3 p0                                = c.read<Transform>(mine).position();
    c.get<CharacterIntent>(mine).desiredVelocity = Vec3(1, 0, 0);
    n.step(2);
    EXPECT_GT(c.read<Transform>(mine).position().x(), p0.x());
}

// A host load with other pins (Resync v2) drops the client's residency: every chunk is requested again with the new
// plans and none is Resident before its new snapshot
TEST(ChunkSync, PinAdoptionDropsResidency)
{
    std::vector<uint8_t> blob;
    {
        // A v1 world with the client's player on C (owner peer 1, saved)
        static std::shared_ptr<ContentCache> cache = std::make_shared<ContentCache>();
        World v1(worldNetTestSetup(true, 1, cache));
        clientPlayer(v1, centreOf(kC));
        blob = v1.save();
    }
    NetPeers n(worldNetTestSetup(true, 2));
    World& h = *n.host;
    World& c = n.client();
    ASSERT_TRUE(n.stepUntil([&] { return c.joinState() == JoinState::Joined; }));
    clientPlayer(h, centreOf(kC));
    ASSERT_TRUE(n.stepUntil([&] { return c.residency().state(kC) == ChunkState::Resident; }));
    ASSERT_EQ(c.generation().space(0).pinned, 2u);
    ASSERT_TRUE(h.tryLoad(blob).ok());
    bool residentBeforeSnapshot = false, dropped = false;
    for (int k = 0; k < 64; ++k)
    {
        n.step();
        if (c.replication().epoch() == 1 && c.residency().state(kC) != ChunkState::Resident) dropped = true;
        if (c.residency().state(kC) == ChunkState::Resident && !c.chunkSynced(kC)) residentBeforeSnapshot = true;
        if (dropped && c.residency().state(kC) == ChunkState::Resident) break;
    }
    EXPECT_EQ(c.generation().space(0).pinned, 1u);
    EXPECT_TRUE(dropped);
    EXPECT_FALSE(residentBeforeSnapshot);
    EXPECT_EQ(c.residency().state(kC), ChunkState::Resident);
    expectChunkEqual(h, c, kC);
}

// A gap on a Resident chunk re-requests it: the chunk stays Resident (the mover does not hold) and ends equal to the
// host after the new snapshot
TEST(ChunkSync, RerequestKeepsResident)
{
    NetPeers n(worldNetTestSetup(true));
    World& h = *n.host;
    World& c = n.client();
    ASSERT_TRUE(n.stepUntil([&] { return c.joinState() == JoinState::Joined; }));
    clientPlayer(h, centreOf(kC));
    ASSERT_TRUE(n.stepUntil([&] { return c.residency().state(kC) == ChunkState::Resident; }));
    h.editTerrain(0, raiseBrush(centreOf(kC).x() + 30, centreOf(kC).z()));
    ASSERT_TRUE(n.stepUntil([&] { return c.terrainEdits().sequence(kC) == h.terrainEdits().sequence(kC); }));
    const uint64_t holds = c.residency().stats().moverHolds;
    const std::vector<uint8_t> m =
        modificationBytes(c.replication().epoch(), 0, {partOf(kC, h.terrainEdits().sequence(kC) + 2, 3, 3, 1, 1, 5)});
    c.replication().chunkSync().onTerrainModification(0, m.data(), m.size());
    ASSERT_EQ(c.replication().chunkSync().state(kC), ClientChunkSync::State::Pending);
    for (int k = 0; k < 8 && !c.chunkSynced(kC); ++k)
    {
        n.step();
        EXPECT_EQ(c.residency().state(kC), ChunkState::Resident);
        EXPECT_TRUE(c.chunkReady(0, centreOf(kC).x(), centreOf(kC).z()));
    }
    EXPECT_TRUE(c.chunkSynced(kC));
    EXPECT_EQ(c.residency().stats().moverHolds, holds);
    EXPECT_EQ(c.netStats().chunkRerequests, 1u);
    expectChunkEqual(h, c, kC);
}

// The subscription starts at the request (prefetch): the add is sent in the step the residency requests the chunk,
// before its base is generated, and the snapshot is applied before the chunk becomes resident
TEST(ChunkSync, SubscribeAtRequest)
{
    WorldSetup setup = worldNetTestSetup(true);
    for (SpaceTemplate& t : setup.config.spaceTemplates) t.prefetchMargin = 256;
    NetPeers n(setup);
    World& h = *n.host;
    World& c = n.client();
    // An asynchronous client residency whose jobs wait for a latch
    c.setStreamingSynchronous(false);
    std::atomic<bool> release{false};
    c.residency().jobHookForTests = [&](const ChunkId&)
    {
        while (!release) std::this_thread::yield();
    };
    ASSERT_TRUE(n.stepUntil([&] { return c.joinState() == JoinState::Joined; }));
    clientPlayer(h, centreOf(kC));
    ASSERT_TRUE(n.stepUntil([&] { return c.residency().requested(kC); }));
    // Requested, not resident (the job waits), already subscribed and synced
    EXPECT_EQ(c.residency().state(kC), ChunkState::Requested);
    EXPECT_NE(c.replication().chunkSync().state(kC), ClientChunkSync::State::None);
    ASSERT_TRUE(n.stepUntil([&] { return c.chunkSynced(kC); }));
    EXPECT_EQ(c.residency().state(kC), ChunkState::Requested);
    release = true;
    ASSERT_TRUE(n.stepUntil([&] { return c.residency().state(kC) == ChunkState::Resident; }, 200));
    c.residency().jobHookForTests = nullptr;
}

// An arrival 3 km away: the replicated PendingTransition makes the target a client centre, the client subscribes the
// target's chunks before the host moves the player; then the player is held until its chunk is synced
TEST(ChunkSync, ArrivalSubscribesTarget)
{
    NetPeers n(worldNetTestSetup(true));
    World& h = *n.host;
    World& c = n.client();
    ASSERT_TRUE(n.stepUntil([&] { return c.joinState() == JoinState::Joined; }));
    const Entity player = clientPlayer(h, Vec3(300, 0, 300));
    ASSERT_TRUE(n.stepUntil([&] { return c.replication().entityOfNetId(h.replication().netIdOf(player)).valid(); }));
    ASSERT_TRUE(n.stepUntil([&] { return c.chunkSynced(chunkAt(0, 1, 1)); }));
    const Vec3 target(3300, 0, 3300);
    const ChunkId far = chunkAt(0, 12, 12);
    // The host loads the target island asynchronously and its jobs wait for a latch: the arrival completes only after
    // the release (synchronous worlds complete it in the next step)
    std::atomic<bool> release{false};
    h.setStreamingSynchronous(false);
    h.residency().jobHookForTests = [&](const ChunkId&)
    {
        while (!release) std::this_thread::yield();
    };
    ASSERT_TRUE(h.requestArrival(player, 0, target));
    bool subscribedBeforeMove = false;
    for (int k = 0; k < 64; ++k)
    {
        if (k == 8) release = true;
        n.step();
        const Entity mine = c.replication().entityOfNetId(h.replication().netIdOf(player));
        const bool moved  = mine.valid() && (c.read<Transform>(mine).position() - target).norm() < 300;
        if (!moved && c.replication().chunkSync().state(far) != ClientChunkSync::State::None)
            subscribedBeforeMove = true;
        if (moved && c.chunkSynced(far) && c.residency().state(far) == ChunkState::Resident) break;
    }
    release                       = true;
    h.residency().jobHookForTests = nullptr;
    EXPECT_TRUE(subscribedBeforeMove);
    EXPECT_TRUE(c.chunkSynced(far));
    EXPECT_EQ(c.residency().state(far), ChunkState::Resident);
}

// D19: a world without streaming subscribes everything: the index lists the chunks with state, an edit of an unlisted
// chunk in flight is applied from sequence 1, holds are no-ops, removes and adds outside an `all` space are rejected
TEST(ChunkSync, AllSubscriptionWithoutStreaming)
{
    NetPeers n(worldNetTestSetup(false));
    World& h = *n.host;
    World& c = n.client();
    std::vector<ChunkId> edited;
    for (int32_t x = 1; x <= 3; ++x)
    {
        edited.push_back(chunkAt(0, x, 1));
        h.editTerrain(0, raiseBrush(centreOf(edited.back()).x(), centreOf(edited.back()).z()));
    }
    const ChunkId recorded = chunkAt(0, 9, 9);
    h.contentRecordsForUpdate().injectForTests(recorded, 2, {InstanceKey{0, 1}});
    h.onRecordChanged(recorded);
    ASSERT_TRUE(n.stepUntil([&] { return c.joinState() == JoinState::Joined; }));
    // Between the index's encoding and its arrival: an edit of an unlisted chunk
    ASSERT_TRUE(n.stepUntil([&] { return h.replication().subscriptions().stats().indexesSent > 0; }));
    const ChunkId later = chunkAt(0, 7, 3);
    h.editTerrain(0, raiseBrush(centreOf(later).x(), centreOf(later).z()));
    ASSERT_TRUE(n.stepUntil([&] { return synced(n); }));
    EXPECT_EQ(h.replication().subscriptions().stats().snapshotsSent, 4u);
    for (const ChunkId& x : edited) expectChunkEqual(h, c, x);
    expectChunkEqual(h, c, recorded);
    expectChunkEqual(h, c, later);
    EXPECT_EQ(c.terrainEdits().chunkCount(), h.terrainEdits().chunkCount());
    // A hold is a no-op
    c.replication().chunkSync().hold(chunkAt(0, 2, 2));
    EXPECT_TRUE(c.replication().chunkSync().holds().empty());
    // Rejected: a remove in a world without streaming, an add in a space the peer has not subscribed as a whole
    HostChunkSubscriptions& subs = h.replication().subscriptions();
    const uint64_t r             = h.netStats().rejected;
    auto send                    = [&](const ChunkInterestMessage& m)
    {
        std::vector<uint8_t> b;
        encodeChunkInterest(m, b);
        subs.onInterest(1, b.data(), b.size());
    };
    ChunkInterestMessage m;
    m.epoch   = h.replication().epoch();
    m.removes = {{1, 1}};
    send(m);
    ChunkInterestMessage other;
    other.epoch = m.epoch;
    other.space = 99;
    other.adds  = {ChunkInterestMessage::Add{1, 1, 5}};
    send(other);  // an unknown space
    EXPECT_EQ(h.netStats().rejected, r + 2);
    // A re-request inside an `all` space is accepted
    ChunkInterestMessage re;
    re.epoch = m.epoch;
    re.adds  = {ChunkInterestMessage::Add{1, 1, 77}};
    send(re);
    EXPECT_EQ(h.netStats().rejected, r + 2);
    EXPECT_EQ(subs.queuedSnapshots(1), 1u);
    EXPECT_EQ(c.netStats().modificationGaps, 0u);
}

// D16: aborted phase 8s keep the protocol: the host keeps its served snapshots queued, a client's staged interest is
// computed again, a re-request and an `all` subscription whose phase 8 aborted are sent next step
TEST(ChunkSync, AbortsKeepTheProtocol)
{
    {
        NetPeers n(worldNetTestSetup(false));
        World& h = *n.host;
        World& c = n.client();
        h.editTerrain(0, raiseBrush(centreOf(kC).x(), centreOf(kC).z()));
        ASSERT_TRUE(n.stepUntil([&] { return c.joinState() == JoinState::Joined; }));
        // The client's first phase 8 after Joined (the `all` subscription) aborts
        bool once               = false;
        c.debugAfterReplication = [&]()
        {
            if (once) return;
            once = true;
            throw std::runtime_error("abort the client's phase 8");
        };
        h.step();
        EXPECT_THROW(c.step(), std::runtime_error);
        c.debugAfterReplication = nullptr;
        // The host's phase 8 aborts after it served the snapshot
        bool hostOnce           = false;
        h.debugAfterReplication = [&]()
        {
            if (hostOnce || h.replication().subscriptions().servedCount() == 0) return;
            hostOnce = true;
            throw std::runtime_error("abort the host's phase 8");
        };
        for (int k = 0; k < 64 && !synced(n); ++k)
        {
            try
            {
                h.step();
            }
            catch (const std::runtime_error&)
            {
            }
            c.step();
        }
        h.debugAfterReplication = nullptr;
        EXPECT_TRUE(hostOnce);
        ASSERT_TRUE(synced(n));
        expectChunkEqual(h, c, kC);
        EXPECT_EQ(c.netStats().modificationGaps, 0u);
    }
    {
        // A streaming client: the staged interest and a re-request of an aborted phase 8 are sent again
        NetPeers n(worldNetTestSetup(true));
        World& h           = *n.host;
        World& c           = n.client();
        ClientChunkSync& s = c.replication().chunkSync();
        ASSERT_TRUE(n.stepUntil([&] { return c.joinState() == JoinState::Joined; }));
        h.editTerrain(0, raiseBrush(centreOf(kC).x(), centreOf(kC).z()));
        s.hold(kC);
        bool once               = false;
        c.debugAfterReplication = [&]()
        {
            if (once) return;
            once = true;
            throw std::runtime_error("abort the client's phase 8");
        };
        h.step();
        EXPECT_THROW(c.step(), std::runtime_error);
        EXPECT_EQ(s.state(kC), ClientChunkSync::State::None);  // the staged add was discarded
        ASSERT_TRUE(n.stepUntil([&] { return c.chunkSynced(kC); }));
        // A gap's re-request whose phase 8 aborts
        const std::vector<uint8_t> m = modificationBytes(
            c.replication().epoch(), 0, {partOf(kC, h.terrainEdits().sequence(kC) + 2, 1, 1, 1, 1, 9)});
        s.onTerrainModification(0, m.data(), m.size());
        once = false;
        h.step();
        EXPECT_THROW(c.step(), std::runtime_error);
        c.debugAfterReplication = nullptr;
        ASSERT_TRUE(n.stepUntil([&] { return c.chunkSynced(kC); }));
        expectChunkEqual(h, c, kC);
    }
}

// Complete points never block on the network (S24 Scope 2): completeResidency returns although C's snapshot is held
namespace
{
bool g_throwClientPhase1 = false;
void throwingClientPhase1(World&, double)
{
    if (!g_throwClientPhase1) return;
    g_throwClientPhase1 = false;
    throw std::runtime_error("abort the client's phase 1");
}
void abortSystems(SystemList& s)
{
    StreamingTest::registerStreamingTestSystems(s);
    s.add(Phase::Input, "throw_client_phase1", throwingClientPhase1, RunOn::Client);
}
}  // namespace

// Regression (review round 1): the client sends its JoinRequest while it receives the Connected event in phase 1; an
// abort of that phase dropped the request from the outbox, the event was consumed, and the client stayed Joining
TEST(Handshake, JoinRequestSurvivesAnAbortedPhase1)
{
    WorldSetup setup = worldNetTestSetup(false);
    setup.systems    = abortSystems;
    NetPeers n(setup);
    World& h            = *n.host;
    World& c            = n.client();
    g_throwClientPhase1 = true;
    h.step();
    EXPECT_THROW(c.step(), std::runtime_error);
    ASSERT_TRUE(n.stepUntil([&] { return c.joinState() == JoinState::Joined; }));
}

TEST(ChunkSync, CompletePointDoesNotBlock)
{
    WorldSetup setup                               = worldNetTestSetup(true);
    setup.config.network.chunkSnapshotBytesPerStep = 1;
    NetPeers n(setup);
    World& h = *n.host;
    World& c = n.client();
    ASSERT_TRUE(n.stepUntil([&] { return c.joinState() == JoinState::Joined; }));
    clientPlayer(h, centreOf(kC));
    ASSERT_TRUE(n.stepUntil([&] { return c.residency().requested(kC); }));
    const auto t0 = std::chrono::steady_clock::now();
    EXPECT_TRUE(c.completeResidency(30));
    EXPECT_LT(std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), 10.0);
    // Unconditional: the gate decides the state (the snapshot may or may not have arrived by now)
    EXPECT_EQ(c.residency().state(kC), c.chunkSynced(kC) ? ChunkState::Resident : ChunkState::Requested);
}

TEST(ChunkSync, ReadinessWaitsForSnapshots)
{
    WorldSetup setup                               = worldNetTestSetup(true);
    setup.config.network.chunkSnapshotBytesPerStep = 1;
    NetPeers n(setup);
    World& c = n.client();
    ASSERT_TRUE(n.stepUntil([&] { return c.joinState() == JoinState::Joined; }));
    // A flat space (terrain and content readiness 1): only the subscriptions count
    const SpaceId flat = StreamingTest::spaceNamed(c, "test_flat");
    ASSERT_EQ(c.spaceReadiness(flat, centreOf(kC), 300), 1.0);
    for (int32_t x = 3; x <= 5; ++x) c.replication().chunkSync().hold(chunkAt(flat, x, 5));
    c.step();
    EXPECT_LT(c.spaceReadiness(flat, centreOf(kC), 300), 1.0);
    // A free camera far away: the subscribed chunks are outside its radius
    EXPECT_EQ(c.spaceReadiness(flat, Vec3(3500, 0, 3500), 300), 1.0);
    ASSERT_TRUE(n.stepUntil([&] { return synced(n); }));
    EXPECT_EQ(c.spaceReadiness(flat, centreOf(kC), 300), 1.0);
}

TEST(TombstoneCap, RefusesBeyondTheCap)
{
    World w(worldNetTestSetup(false));
    std::vector<InstanceKey> keys;
    for (uint32_t k = 0; k < kMaxChunkTombstones; ++k) keys.push_back(InstanceKey{0, k});
    w.contentRecordsForUpdate().injectForTests(kC, w.effectivePlacementVersion(kC), keys);
    InstanceKey extra{1, 0};
    for (const TierOneInstance& t : w.generation().chunkPlacement(kC, w.effectivePlacementVersion(kC))->instances)
        if (t.key.generatorSlot == 1)
        {
            extra = t.key;
            break;
        }
    std::string error;
    EXPECT_FALSE(w.checkTombstone(kC, extra, &error));
    EXPECT_NE(error.find("tombstones (at most"), std::string::npos) << error;
}

}  // namespace
