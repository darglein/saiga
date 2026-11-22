/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The replication of a World (docs/engine_design.md §13.3, §13.4; S16): the host's NetId table and its phase-8
// replication (spawns, destroys, hierarchy and component changes, full-state snapshots), staged and committed with the
// step; the client's NetId map and the handlers of the engine messages (phase 1); owner state, owner commands and
// owner poses; the host clock and the interpolation buffers; the write detector of the scenario variant.

#pragma once

#include "saiga/core/math/Quaternion.h"
#include "saiga/core/math/Types.h"
#include "saiga/engine/ecs/Entity.h"
#include "saiga/engine/ecs/Ids.h"
#include "saiga/engine/gameplay/PlayerAction.h"
#include "saiga/engine/net/HostClock.h"
#include "saiga/engine/net/InterpolationSystem.h"
#include "saiga/engine/net/NetIds.h"
#include "saiga/engine/net/NetVisitors.h"
#include "saiga/engine/terrain/TerrainEdit.h"
#include "saiga/engine/world/Phase.h"
#include "saiga/export.h"

#include <array>
#include <functional>
#include <map>
#include <string>
#include <vector>

#include <unordered_map>

namespace Saiga
{
namespace Engine
{
class World;
struct OwnerCommandMessage;
struct DestroyMessage;
struct HierarchyMessage;
struct PlayerActionMessage;
struct PingMessage;
struct PongMessage;
struct ResyncMessage;

class SAIGA_ENGINE_API Replication final : public NetIdResolver
{
   public:
    // The per-entity bound of the snapshot state (§13.4): a larger entity stops the program, naming it and its largest
    // component (register such a component HostReliable)
    static constexpr size_t kMaxEntitySnapshotBytes = 1100;

    struct Stats
    {
        uint64_t snapshotsSent = 0, snapshotMessagesSent = 0, snapshotBytesSent = 0, compressedMessages = 0;
        uint64_t spawnsSent = 0, destroysSent = 0, hierarchySent = 0, componentChangesSent = 0;
        uint64_t snapshotsApplied = 0, snapshotsDroppedEpoch = 0;
        uint64_t ownerStatesDropped = 0;  // an older command counter or a sender that does not own the entity
        uint64_t actionsDropped     = 0;  // PlayerActions for players the sender does not own
        // S22: terrain messages sent (host) / applied, duplicate parts dropped and messages of another epoch (client)
        uint64_t terrainMessagesSent = 0, terrainBytesSent = 0, terrainMessagesApplied = 0;
        uint64_t terrainPartsDuplicate = 0, terrainDroppedEpoch = 0;
    };

    explicit Replication(World& world);
    ~Replication() override;
    Replication(const Replication&)            = delete;
    Replication& operator=(const Replication&) = delete;

    // ---- NetIds (both roles) ----
    // Host: the NetId of a spawned (or staged) entity; client: of a replicated entity; 0 otherwise
    NetId netIdOf(Entity e) const override;
    // Host: the spawned entity of a NetId; client: the replicated entity; Entity{} otherwise
    Entity entityOfNetId(NetId id) const override;
    // Host: the entity passed a phase-8 replication (its spawn was sent, or would have been with peers)
    bool spawned(Entity e) const;
    // The current epoch (incremented by every host load)
    uint32_t epoch() const { return currentEpoch; }
    // Host: the sequence number of the last snapshot sent; client: of the last snapshot applied
    uint64_t snapshotSequence() const { return snapshotSeq; }
    // Host: the last owner-state sequence received from a peer; client: the last one sent
    uint64_t ownerStateSequence(uint8_t peer) const { return ownerSeq[peer]; }
    // Owner side and host: the command counter of a root (§13.3)
    uint8_t commandCounter(Entity root) const;
    // Host: the snap counter of a root
    uint8_t snapCounter(Entity root) const;
    const HostClock& clock() const { return hostClock; }
    const Stats& stats() const { return counts; }
    // The NetIds of all replicated entities (host: spawned; client: mapped), ascending
    std::vector<NetId> netIds() const;
    // Client: entities that own components of the entity are waiting for an unresolved parent / Entity field
    size_t pendingCount() const { return pendingParents.size() + pendingDecodes.size(); }
    // The interpolation buffer of an entity (nullptr if none)
    InterpolationBuffer* buffer(Entity e);

    // ---- step hooks (called by the World) ----
    // Phase 1 after the handlers: a client applies the newest complete snapshot and updates its clock
    void afterReceive(bool networkOnly);
    // Phase 5 (ownerPoseSystem, host): the owners' latest states of client-owned roots (extrapolated in a real step)
    void ownerPose(bool extrapolate);
    // Phase 8 after the systems and their commands: replication (host), owner state (owners), interpolation
    void extract(bool networkOnly);
    // The step completed: the staged bookkeeping is committed
    void commit();
    // An aborted step: the bookkeeping staged in the aborted phase is dropped (and the log consumer rewound)
    void abort(Phase aborted);
    // A committed host load: a new epoch, every entity is spawned again (Resync to the connected clients)
    void onHostLoad();
    // A client queues an action for an owned player: sent to the host
    void sendPlayerAction(const PlayerAction& action);
    // World::moveToSpace on the host (and ownerTeleport): the move, and for a spawned client-owned root the
    // OwnerCommand to its owner
    void ownerMove(Entity root, SpaceId space, const Vec3& position, const Quat& rotation);
    // The write detector of the scenario variant (§17.3): receivers keep a hash of what they last wrote; a mismatch
    // before the next write (another writer touched a component of a remote authority, R-NET-2) is reported
    void setWriteDetector(std::function<void(const std::string&)> report);

    // ---- message handlers (registerEngineMessages) ----
    void onSpawn(uint8_t from, const uint8_t* data, size_t size);
    void onComponentChange(uint8_t from, const uint8_t* data, size_t size);
    void onSnapshot(uint8_t from, const uint8_t* data, size_t size);
    void onOwnerState(uint8_t from, const uint8_t* data, size_t size);
    void onDestroy(uint8_t from, const DestroyMessage& m);
    void onHierarchy(uint8_t from, const HierarchyMessage& m);
    void onOwnerCommand(uint8_t from, const OwnerCommandMessage& m);
    void onPlayerAction(uint8_t from, const PlayerActionMessage& m);
    void onPing(uint8_t from, const PingMessage& m);
    void onPong(uint8_t from, const PongMessage& m);
    void onResync(uint8_t from, const ResyncMessage& m);
    // S22: a client applies a host's terrain modification / chunk state (validated on its own: no check against the
    // present neighbours, the host is authoritative); no hook is called
    void onTerrainModification(uint8_t from, const uint8_t* data, size_t size);
    void onTerrainChunkState(uint8_t from, const uint8_t* data, size_t size);
    // Host (S22, E5): one TerrainChunkState per chunk with terrain state (sequence > 0) of every space, in SpaceId and
    // ChunkId order (the dense delta only for chunks with deltas): after the spawn set of a join, after Resync on a
    // host load. S26 replaces it with per-loaded-chunk snapshots
    void sendChunkSnapshot(uint8_t peer);

    // ---- the host state list (S22, §13.2): ordered host state messages, unencoded, sent by extract to every regular
    // peer after the component changes (also by stepNetworkOnly's extract), the sent ones removed when the step commits
    // (also without peers; entries appended after the send stay for the next extract), kept by an aborted step (the
    // edits of its finished phases stay applied). S23b appends its tombstones, so the messages that bump one chunk's
    // sequence reach the client in application order
    struct HostStateEntry
    {
        enum class Kind : uint8_t
        {
            TerrainModification
        };
        Kind kind     = Kind::TerrainModification;
        SpaceId space = 0;
        std::vector<TerrainPart> parts;
    };
    // Reserve: capacity for one more entry (the edit's reserve step; the append then allocates nothing). Grows
    // geometrically (an exact reserve would reallocate on every edit of a batch)
    void reserveHostState()
    {
        if (hostStates.size() == hostStates.capacity()) hostStates.reserve(2 * hostStates.size() + 1);
    }
    // Host: one TerrainModification (applyTerrainEdit; after reserveHostState)
    void appendTerrainModification(SpaceId space, std::vector<TerrainPart>&& parts);
    const std::vector<HostStateEntry>& hostStateList() const { return hostStates; }
    // A committed host load: the entries describe the replaced state (the load's snapshot follows from onHostLoad)
    void clearHostStates()
    {
        hostStates.clear();
        hostStatesSent = 0;
    }

    // The component encodings (engine-internal; also the replica comparison and the benchmarks)
    void encodeComponent(Entity e, ComponentId id, std::vector<uint8_t>& out) const;
    std::vector<NetLeaf> flattenComponent(Entity e, ComponentId id) const;

   private:
    // ---- host ----
    struct HostSlot
    {
        uint32_t generation    = 0;
        NetId netId            = 0;
        bool spawned           = false;
        uint8_t owner          = 0;
        uint8_t snap           = 0;             // bumped by a teleport of the root (sent in snapshots)
        uint8_t command        = 0;             // incremented by ownerMove of a client-owned root
        uint64_t snapCountedAt = ~uint64_t(0);  // the step whose teleport was counted
    };
    struct SlotUndo
    {
        Phase phase;
        uint32_t index;
        HostSlot old;
    };
    struct HashUndo
    {
        Phase phase;
        uint64_t key;
        bool existed;
        uint64_t old;
    };
    struct NetIdUndo
    {
        Phase phase;
        NetId id;
        bool existed;
        Entity old;
    };
    HostSlot* hostSlot(Entity e);
    const HostSlot* hostSlot(Entity e) const;
    HostSlot& writeSlot(uint32_t index);  // journals the old value during a step
    void setHash(uint64_t key, uint64_t value);
    void eraseHash(uint64_t key);
    void setNetIdEntry(NetId id, Entity e);  // Entity{}: erase
    bool hostReplicated(Entity e) const;     // created on the host, not NetLocal (itself or an ancestor)
    int depthOf(Entity e) const;
    void hostReplicate(bool networkOnly);
    void encodeSpawn(Entity e, std::vector<uint8_t>& out);
    void encodeRecord(Entity e, const HostSlot& s, bool withOwnerAuthoritative, std::vector<uint8_t>& out);
    void sendSnapshot(const std::vector<uint8_t>& peers);
    void sendToAll(const std::vector<uint8_t>& peers, MessageId id, const std::vector<uint8_t>& payload);
    std::vector<uint8_t> regularPeers, joinedPeers;
    std::vector<HostStateEntry> hostStates;
    // The leading entries of hostStates that this step's extract sent (or had no peer for): commit removes only those,
    // an entry appended after the send (an edit applied by the flush after stepNetworkOnly's extract) waits for the
    // next
    size_t hostStatesSent = 0;
    // Phase 8 (both extract paths): the host state list to the regular peers
    void sendHostStates();

    std::vector<HostSlot> host;
    std::unordered_map<NetId, Entity> hostByNetId;
    std::unordered_map<uint64_t, uint64_t> reliableHash;  // (index << 8 | component) -> hash of the last sent bytes
    NetId nextNetId = kFirstDynamicNetId;
    int logConsumer = -1;
    std::vector<SlotUndo> slotUndo;
    std::vector<HashUndo> hashUndo;
    std::vector<NetIdUndo> netIdUndo;
    bool replicatedThisStep    = false;
    uint64_t logCursorBefore   = 0;
    NetId nextNetIdBefore      = 0;
    uint64_t snapshotSeqBefore = 0;
    // Scratch of the replication (kept across steps: no allocation once warm)
    std::vector<Entity> candidates, destroyedNow, hierarchyNow;
    std::vector<std::pair<Entity, ComponentId>> changedNow;
    std::vector<uint8_t> scratch, record;
    void* zstd = nullptr;  // the compression context of the snapshots

    // The latest owner states of client-owned entities (host), by entity index
    struct OwnerRecord
    {
        uint32_t generation = 0;
        uint8_t peer        = 0;
        bool havePose = false, havePrevious = false;
        double stamp = 0, previousStamp = 0;
        Vec3 position = Vec3::Zero(), previousPosition = Vec3::Zero();
        Quat rotation = Quat::Identity();
        Vec3 scale    = Vec3::Ones();
        // The owner-authoritative components other than Transform: (component, bytes)
        std::vector<std::pair<ComponentId, std::vector<uint8_t>>> components;
    };
    std::map<uint32_t, OwnerRecord> ownerRecords;
    // ownerMove's changes of the owner record and the interpolation buffer, undone by an abort of their phase
    struct TeleportUndo
    {
        Phase phase;
        uint32_t index;
        bool hadRecord;
        OwnerRecord record;
        bool hadBuffer;
        std::pair<uint32_t, InterpolationBuffer> buffer;
        // The root's pose and space before the move: restored with the command counter, so host and owner agree
        uint32_t generation = 0;
        SpaceId space       = 0;
        Vec3 position       = Vec3::Zero();
        Quat rotation       = Quat::Identity();
    };
    std::vector<TeleportUndo> teleportUndo;

    // ---- client ----
    struct ClientEntry
    {
        Entity entity;
        uint8_t snap    = 0;
        uint8_t command = 0;
    };
    std::unordered_map<NetId, ClientEntry> clientByNetId;
    std::vector<std::pair<uint32_t, NetId>> clientNetIdOf;  // by entity index: (generation, NetId)
    void mapClient(NetId id, Entity e, uint8_t snap, uint8_t command);
    void unmapClient(Entity e);
    ClientEntry* clientEntry(Entity e);
    // Waiting for a parent that has not arrived (child NetId -> parent NetId)
    std::map<NetId, NetId> pendingParents;
    // Components with an unresolved Entity field, decoded again when a spawn arrives
    struct PendingDecode
    {
        NetId netId;
        ComponentId component;
        std::vector<uint8_t> bytes;
    };
    std::vector<PendingDecode> pendingDecodes;
    void retryPending();
    // A component from the wire into the entity (added if missing); false on malformed input
    bool applyComponent(Entity e, ComponentId id, const uint8_t* data, size_t size, bool rememberUnresolved,
                        NetId netId);
    void applyTransform(Entity e, const uint8_t* data, size_t size, bool teleport, bool track);
    void destroyReplicated(Entity e, bool callbacks);
    struct SnapshotParts
    {
        uint64_t hostStep = 0;
        uint32_t parts    = 0;
        std::vector<std::vector<uint8_t>> bodies;  // by part (empty: missing)
        uint32_t received = 0;
    };
    std::map<uint64_t, SnapshotParts> pendingSnapshots;
    uint64_t newestComplete = 0;
    void applySnapshot(const SnapshotParts& s);
    void ownerStateSend(bool networkOnly);

    // ---- write detector ----
    std::function<void(const std::string&)> detector;
    std::unordered_map<uint64_t, std::pair<uint32_t, uint64_t>> written;  // (index << 8 | id) -> (generation, hash)
    uint64_t currentHash(Entity e, ComponentId id) const;
    // Before a receiver overwrites: a mismatch is reported
    void checkWritten(Entity e, ComponentId id, const char* receiver);
    // After any legitimate write
    void noteWritten(Entity e, ComponentId id);
    std::string describe(Entity e) const;

    World& w;
    uint32_t currentEpoch = 0;
    uint64_t snapshotSeq  = 0;
    std::array<uint64_t, 256> ownerSeq{};
    HostClock hostClock;
    std::unordered_map<uint32_t, std::pair<uint32_t, InterpolationBuffer>> buffers;  // index -> (generation, buffer)
    Stats counts;
    friend SAIGA_ENGINE_API void interpolationSystem(World& world, double dt);
    // The space primitive (S21) notes its writes (World::setSubtreeSpace, teleportTo)
    friend class World;
};

// Phase 5 (host): the owners' latest states of client-owned roots (§13.3); see Replication::ownerPose
SAIGA_ENGINE_API void ownerPoseSystem(World& world, double dt);

}  // namespace Engine
}  // namespace Saiga
