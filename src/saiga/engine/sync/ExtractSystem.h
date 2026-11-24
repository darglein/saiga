/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The extract (docs/engine_design.md §10.3; S19b), owned by the World. Reads: RenderInstance, Transform and its change
// flags, RenderPose and its flags, Parent, LocalTransform, the structural log (its own consumer), the owner table.
// Writes: no components; the render sink's dynamic instances (prev / curr pose pairs relative to a chunk).
//
// Per step (World::runStep, after the interpolation at the end of phase 8): the structural log creates and frees the
// render instances (a creation sends the current pose as a snap); the entities whose pose changed in this or the
// previous step get the pair (prev = old curr, curr = new; a teleport: prev = curr). The pairs commit with the step
// (an aborted step does not shift them). Between steps catchUp() sends edits (a paused world, tools, loads) as snaps.

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/engine/components/LightSource.h"
#include "saiga/engine/ecs/Entity.h"
#include "saiga/engine/sync/RenderSink.h"
#include "saiga/export.h"

#include <cstdint>
#include <vector>

#include <unordered_set>

namespace Saiga
{
namespace Engine
{
class World;

// A pose in world space (double; cameras, cosmetic instances)
struct WorldPose
{
    SpaceId space = 0;
    Vec3 position = Vec3::Zero();
    Quat rotation = Quat::Identity();
    double scale  = 1;
};

// A world pose relative to the chunk that contains it (RenderSink::setDynamic)
struct ChunkPose
{
    ChunkId chunk;
    DynamicPose pose;
};
SAIGA_ENGINE_API ChunkPose toChunkPose(const WorldPose& p);

class SAIGA_ENGINE_API ExtractSystem
{
   public:
    explicit ExtractSystem(World& world);
    ~ExtractSystem();
    ExtractSystem(const ExtractSystem&)            = delete;
    ExtractSystem& operator=(const ExtractSystem&) = delete;

    // ---- readers (between steps) ----

    // The pose of a tracked entity at alpha between its last two extracted poses (S18's interpolatePose in double, the
    // GPU's formula), in world space. False for an untracked entity (no sink, dead, spawned since the last extract or
    // catch-up, without RenderInstance, a remote entity before its first RenderPose); out is untouched then.
    // teleported: the current pair is a snap (the same for every reader until the next pair).
    bool interpolatedPose(Entity e, double alpha, WorldPose& out, bool* teleported = nullptr) const;
    // The sequence number of the last snap pair of a tracked entity (its creation, teleports, source switches, edits
    // sent by the catch-up; unique in this World, so a re-created instance after a load gets a new one; 0: untracked):
    // it changes once per snap and stays the same while the entity is idle, so a reader that keeps the last value
    // reacts once (a chase camera's reset), unlike the level of interpolatedPose's teleported
    uint64_t snapSequence(Entity e) const;
    // Between steps (the app before rendering, tools that render): when markDirty() was called since the last step, or
    // when no step completed since the last call and (the world is paused, its time scale is 0, or the structural log
    // has unread records), the log is consumed and every tracked entity whose source pose differs bitwise from the last
    // extracted one (or with a pending teleport) is sent as a snap. Returns whether it ran.
    bool catchUp();
    // An edit between steps (a world API operation, a load): the next catchUp runs, unless a step runs first (its
    // extract sends the edit)
    void markDirty() { dirty = true; }

    // ---- local lights (S36b, §10.7) ----

    // Every frame (the app, after catchUp): the lights of moving entities follow their interpolated pose (setLight with
    // the chunk-relative position; a chunk change is a new chunk). The steps (and catch-ups) send the lights at the
    // current pose already: a world without frames (a scenario) has its lights at the step's pose
    void updateLights(double alpha);
    // The live lights (an entity with a LightSource that is on, valid and has a pose) and the sink light of an entity
    // (invalid: none)
    size_t lightCount() const;
    LightHandle lightOf(Entity e) const;
    // The entity of each sink light (the Lighting tab, list_lights)
    std::vector<std::pair<Entity, LightHandle>> lightEntities() const;
    // Every tracked light entity, also those without a sink light (round 1, finding 8: a light switched off must stay
    // visible and switchable in the Lighting tab and list_lights): the sink light (invalid: none), the component as
    // last read, the entity's pose (when it has one) and the state: "rendered", "off", "invalid: <reason>" or
    // "no pose"
    struct LightEntityState
    {
        Entity entity;
        LightHandle handle;
        LightSource source;
        bool hasPose = false;
        WorldPose pose;
        std::string state;
    };
    std::vector<LightEntityState> lightStates() const;

    // Tracked entities (with RenderInstance) and those with a sink instance
    size_t trackedCount() const { return tracked.size(); }
    size_t instanceCount() const;
    // Entries of the list of non-Transform sources evaluated every step (tests: one per entity)
    size_t remoteQueued() const { return remote.size(); }
    // The sink instance of an entity (invalid: none)
    InstanceHandle instanceOf(Entity e) const;
    RenderSink* sink() const { return renderSink; }

    // ---- World (engine-internal) ----

    // Replaces the sink: the old one's instances are removed and the consumer dropped; a new one gets a consumer and
    // instances for the existing entities (snaps). null: nothing is tracked
    void setSink(RenderSink* sink);
    // The step's extract (end of phase 8)
    void run();
    // The step completed (after the transform flags were shifted): the staged pairs become current
    void commit();
    // The step was aborted: the staged pairs are dropped
    void abort();
    // A mesh was registered after entities named it (World::registerMesh with a sink; the unresolved name is kept):
    // their instances are created
    void meshRegistered();

   private:
    enum class Source : uint8_t
    {
        Transform,   // Transform
        RenderPose,  // the entity's RenderPose
        Composed,    // an ancestor's RenderPose with the local chain below it
        Pending,     // a remote entity (or its child) before its first RenderPose: tracked, no pose
    };
    struct Tracked
    {
        Entity entity;
        InstanceHandle handle;
        MeshId mesh          = 0;
        Source source        = Source::Transform;
        bool hasPose         = false;  // prev / curr are valid (not Pending)
        bool snap            = false;  // the current pair is a snap
        bool teleportHandled = false;  // catch-up handled the pending Transform teleport
        WorldPose prev, curr;
        // The source pose of the last extract or catch-up (the bitwise baseline of catchUp)
        WorldPose seen;
        uint64_t processedRun    = 0;             // the extract or catch-up run that handled it
        uint64_t snapStep        = ~uint64_t(0);  // the step whose Transform teleport the extract handled
        uint64_t snaps           = 0;             // the sequence number of the last snap (snapSequence)
        bool posePending         = false;         // the instance's creation snap was not sent (setDynamic threw)
        bool syncPending         = false;         // on the retry list until the instance exists with its creation snap
        uint8_t handledTeleports = 0;             // Transform::teleportCount of the teleport the catch-up handled
    };
    struct Staged
    {
        Entity entity;
        WorldPose prev, curr;
        bool snap;
        Source source;
        uint64_t snapStep;
    };

    // A light entity (S36b): the component as last sent, the entity's pose pair of the steps, the sink light and the
    // chunk and data last sent
    struct TrackedLight
    {
        Entity entity;
        LightHandle handle;
        LightSource source;
        bool hasPose = false;
        WorldPose prev, curr;
        ChunkId chunk;
        Vulkan::LightData data;
        bool invalidLogged = false;
    };
    std::vector<TrackedLight> lights;
    std::vector<int32_t> lightSlot;  // by entity index: index into lights, -1
    TrackedLight* findLight(Entity e);
    const TrackedLight* findLight(Entity e) const;
    void trackLight(Entity e);
    void untrackLight(Entity e);
    // The poses of the lights after a step (step: the pair shifts; else a snap) and the sends of changed lights
    void syncLights(bool step);
    // Sends the light at alpha between its pose pair (adds, moves or removes the sink light as needed)
    void sendLight(TrackedLight& l, double alpha);

    void attachConsumer();
    void detachConsumer();
    void consumeLog();
    void track(Entity e);
    void untrack(Entity e);
    Tracked* find(Entity e);
    const Tracked* find(Entity e) const;
    // The source of an entity and its pose (false: Pending); root: the entity whose RenderPose is used
    Source sourceOf(Entity e, Entity& renderPoseOwner) const;
    bool sourcePose(Entity e, Source& source, WorldPose& pose) const;
    // Creates, replaces or frees the sink instance of t (mesh name, registered mesh, pose)
    void syncInstance(Tracked& t);
    void send(const Tracked& t);
    void sendPair(InstanceHandle h, const WorldPose& prev, const WorldPose& curr);
    // A tracked entity in this step's extract: the new pair (staged)
    void processStep(Tracked& t, uint64_t step);

    World& w;
    // Entities whose non-uniform scale was logged (once each; the extract uses scale.x)
    mutable std::unordered_set<uint64_t> scaleLogged;
    RenderSink* renderSink = nullptr;
    int consumer           = -1;
    bool retained          = false;
    std::vector<Tracked> tracked;
    std::vector<int32_t> slotOf;  // by entity index: index into tracked, -1
    // Tracked entities whose source is not Transform (checked every step: RenderPose flags, Pending)
    std::vector<Entity> remote;
    std::vector<Staged> staged;
    // Processed by the next step's extract (filled at the commit; an aborted step keeps it): the entities whose
    // committed pair is not yet (curr, curr) and the phase 7-8 movers the step's extract did not see. Without it a late
    // mover's pair (old, new) stays: it is in phase 6's list of the next step only
    std::vector<Entity> settle;
    // Instances whose creation did not complete (addDynamic or the creation snap's setDynamic threw): created and sent
    // again by the next extract or catch-up (an idle entity would otherwise never get an instance or a pose)
    std::vector<Entity> unposed;
    // Handles of untracked entities whose removeDynamic threw: removed again by the next extract or catch-up
    std::vector<InstanceHandle> removals;
    // A RenderPose, Parent or NetOwner record was consumed and the sources are not evaluated again yet
    bool sourcesDirty = false;
    void retryPending();
    // The body of catchUp (after its trigger)
    void catchUpRun();
    void queueRemote(Tracked& t);
    // The entities in remote (index and generation; one entry each)
    std::unordered_set<uint64_t> inRemote;
    static uint64_t entityKey(Entity e) { return (uint64_t(e.index) << 32) | e.generation; }
    bool dirty                      = false;
    uint64_t runId                  = 0;  // counts extract and catch-up runs
    uint64_t snapSeq                = 0;  // the last snap sequence number
    uint64_t committedSteps         = 0;
    uint64_t committedAtLastCatchUp = 0;
};

}  // namespace Engine
}  // namespace Saiga
