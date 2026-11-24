/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// ExtractSystem (§10.3; S19b). Reads: RenderInstance, Transform and its flags, RenderPose and its flags, Parent,
// LocalTransform, the structural log, the owner table. Writes: the render sink's dynamic instances (no components).

#include "saiga/engine/sync/ExtractSystem.h"

#include "saiga/engine/components/LightSource.h"
#include "saiga/engine/components/LocalTransform.h"
#include "saiga/engine/components/Parent.h"
#include "saiga/engine/components/Persistent.h"
#include "saiga/engine/components/RenderInstance.h"
#include "saiga/engine/components/RenderPose.h"
#include "saiga/engine/components/Transform.h"
#include "saiga/engine/ecs/EntityLog.h"
#include "saiga/engine/net/NetOwner.h"
#include "saiga/engine/net/Replication.h"
#include "saiga/engine/world/World.h"

#include <cmath>
#include <cstring>
#include <exception>
#include <limits>

namespace Saiga
{
namespace Engine
{
namespace
{
bool sameBits(double a, double b)
{
    return std::memcmp(&a, &b, sizeof(double)) == 0;
}
// Bitwise (as Transform's setters): equal bits never count as a change
bool samePose(const WorldPose& a, const WorldPose& b)
{
    return a.space == b.space && sameBits(a.position(0), b.position(0)) && sameBits(a.position(1), b.position(1)) &&
           sameBits(a.position(2), b.position(2)) && sameBits(a.rotation.w(), b.rotation.w()) &&
           sameBits(a.rotation.x(), b.rotation.x()) && sameBits(a.rotation.y(), b.rotation.y()) &&
           sameBits(a.rotation.z(), b.rotation.z()) && sameBits(a.scale, b.scale);
}
DynamicPose dynamicPose(const WorldPose& p, const Vec3& origin)
{
    DynamicPose d;
    d.position = p.position - origin;
    d.rotation = p.rotation;
    d.scale    = p.scale;
    return d;
}
}  // namespace

ChunkPose toChunkPose(const WorldPose& p)
{
    ChunkPose c;
    c.chunk.space = p.space;
    // Clamped to the int32 chunk range (beyond +-5.5e11 m the conversion would be undefined; the pose is then relative
    // to the last chunk)
    auto chunkOf = [](double x)
    {
        const double c = std::floor(x / chunkSize);
        if (!(c >= double(std::numeric_limits<int32_t>::min()))) return std::numeric_limits<int32_t>::min();
        if (c > double(std::numeric_limits<int32_t>::max())) return std::numeric_limits<int32_t>::max();
        return int32_t(c);
    };
    c.chunk.x = chunkOf(p.position(0));
    c.chunk.z = chunkOf(p.position(2));
    c.pose    = dynamicPose(p, chunkOrigin(c.chunk));
    return c;
}

namespace
{
bool sameLightData(const Vulkan::LightData& a, const Vulkan::LightData& b)
{
    return a.kind == b.kind && a.position == b.position && a.direction == b.direction && a.color == b.color &&
           a.lumen == b.lumen && a.range == b.range && a.cosInner == b.cosInner && a.cosOuter == b.cosOuter &&
           a.castShadows == b.castShadows && a.priority == b.priority && a.flicker == b.flicker;
}
}  // namespace

ExtractSystem::ExtractSystem(World& world) : w(world) {}

ExtractSystem::~ExtractSystem()
{
    // The sink may be gone already (the World is destroyed last): no sink calls; only the consumer is dropped
    if (consumer >= 0) w.structuralLog().removeConsumer(consumer);
    if (retained) w.releaseStepChanges();
}

void ExtractSystem::attachConsumer()
{
    consumer = w.structuralLog().addConsumer();
    // The entities changed in phases 7-8 (after the phase-6 list) are read at the commit (changedDuringLastStep)
    w.retainStepChanges();
    retained = true;
}

void ExtractSystem::detachConsumer()
{
    if (consumer >= 0) w.structuralLog().removeConsumer(consumer);
    consumer = -1;
    if (retained) w.releaseStepChanges();
    retained = false;
}

void ExtractSystem::setSink(RenderSink* sink)
{
    if (renderSink)
    {
        // The old sink's instances go with it
        // (each handle is dropped once removed: a throwing sink leaves only the rest for a retried setSink)
        for (Tracked& t : tracked)
            if (t.handle)
            {
                renderSink->removeDynamic(t.handle);
                t.handle = InstanceHandle();
            }
        while (!removals.empty())
        {
            renderSink->removeDynamic(removals.back());
            removals.pop_back();
        }
        // S36b: the lights go with the sink as well
        for (TrackedLight& l : lights)
            if (l.handle)
            {
                renderSink->removeLight(l.handle);
                l.handle = LightHandle();
            }
        lights.clear();
        lightSlot.clear();
        tracked.clear();
        slotOf.clear();
        remote.clear();
        staged.clear();
        settle.clear();
        unposed.clear();
        inRemote.clear();
        sourcesDirty = false;
        detachConsumer();
    }
    renderSink = sink;
    if (!renderSink) return;
    attachConsumer();
    // The existing entities (the records before the consumer existed are not read): snaps
    std::vector<Entity> existing;
    w.registry().view<const RenderInstance>().each([&](Entity e, const RenderInstance&) { existing.push_back(e); });
    // Every one is tracked also when the sink throws for one (its instance is retried by the next extract or catch-up;
    // the consumer starts after their records, so nothing else would track the rest)
    std::exception_ptr failure;
    for (Entity e : existing)
    {
        try
        {
            track(e);
        }
        catch (...)
        {
            if (!failure) failure = std::current_exception();
        }
    }
    if (failure) std::rethrow_exception(failure);
    // S36b: the existing light entities
    std::vector<Entity> lit;
    w.registry().view<const LightSource>().each([&](Entity e, const LightSource&) { lit.push_back(e); });
    for (Entity e : lit) trackLight(e);
}

void ExtractSystem::meshRegistered()
{
    for (Tracked& t : tracked)
        if (!t.handle) syncInstance(t);
}

size_t ExtractSystem::instanceCount() const
{
    size_t n = 0;
    for (const Tracked& t : tracked) n += t.handle ? 1 : 0;
    return n;
}

InstanceHandle ExtractSystem::instanceOf(Entity e) const
{
    const Tracked* t = find(e);
    return t ? t->handle : InstanceHandle();
}

ExtractSystem::Tracked* ExtractSystem::find(Entity e)
{
    if (e.index >= slotOf.size() || slotOf[e.index] < 0) return nullptr;
    Tracked& t = tracked[size_t(slotOf[e.index])];
    return t.entity == e ? &t : nullptr;
}

const ExtractSystem::Tracked* ExtractSystem::find(Entity e) const
{
    return const_cast<ExtractSystem*>(this)->find(e);
}

ExtractSystem::Source ExtractSystem::sourceOf(Entity e, Entity& renderPoseOwner) const
{
    const Registry& reg = w.registry();
    renderPoseOwner     = Entity{};
    if (reg.registered(RenderPose::id) && reg.has<RenderPose>(e))
    {
        renderPoseOwner = e;
        return Source::RenderPose;
    }
    // Up the hierarchy: the first ancestor with a RenderPose, else the root (bounded by the slot count: no cycles)
    Entity a             = e;
    const uint32_t limit = reg.slotCount();
    for (uint32_t n = 0; n <= limit; ++n)
    {
        const Parent* p = reg.tryRead<Parent>(a);
        if (!p || !reg.alive(p->entity)) break;
        a = p->entity;
        if (reg.registered(RenderPose::id) && reg.has<RenderPose>(a))
        {
            renderPoseOwner = a;
            return Source::Composed;
        }
    }
    // A remote root (replicated and owned by another peer: on a client every replicated root it does not own, on the
    // host the client-owned roots) is drawn from its RenderPose only (no pop back by the interpolation delay)
    if (w.ownerOf(a) != w.localPeer() && (w.isHost() || w.replication().netIdOf(a) != 0)) return Source::Pending;
    return Source::Transform;
}

bool ExtractSystem::sourcePose(Entity e, Source& source, WorldPose& pose) const
{
    const Registry& reg = w.registry();
    const Transform& t  = reg.read<Transform>(e);
    pose.space          = t.space();
    // Uniform scale: x (a non-uniform scale is rejected by the world API and logged once per entity for game code)
    pose.scale = t.scale()(0);
    if (!sameBits(t.scale()(1), pose.scale) || !sameBits(t.scale()(2), pose.scale))
    {
        if (scaleLogged.insert((uint64_t(e.index) << 32) | e.generation).second)
            SAIGA_LOG_E("extract", e, "has the non-uniform scale (%g, %g, %g); it is rendered with %g", t.scale()(0),
                        t.scale()(1), t.scale()(2), pose.scale);
    }
    Entity owner;
    source = sourceOf(e, owner);
    switch (source)
    {
        case Source::Transform:
            pose.position = t.position();
            pose.rotation = t.rotation();
            return true;
        case Source::RenderPose:
        {
            const RenderPose& rp = reg.read<RenderPose>(e);
            pose.position        = rp.position;
            pose.rotation        = rp.rotation;
            pose.space           = rp.space;  // S21: the interpolated space, not the Transform's
            return true;
        }
        case Source::Composed:
        {
            // The local chain from the RenderPose ancestor down to e (as interpolationSystem composes a child)
            std::vector<Entity> chain;
            for (Entity a = e; a != owner; a = reg.read<Parent>(a).entity) chain.push_back(a);
            const RenderPose& rp = reg.read<RenderPose>(owner);
            Vec3 p               = rp.position;
            Quat q               = rp.rotation;
            Vec3 parentScale     = reg.read<Transform>(owner).scale();
            for (size_t i = chain.size(); i-- > 0;)
            {
                const LocalTransform* local = reg.tryRead<LocalTransform>(chain[i]);
                if (local)
                {
                    p = p + q * parentScale.cwiseProduct(local->position);
                    q = q * local->rotation;
                }
                parentScale = reg.read<Transform>(chain[i]).scale();
            }
            pose.position = p;
            pose.rotation = q;
            pose.space    = rp.space;  // the RenderPose owner's space
            return true;
        }
        case Source::Pending:
            return false;
    }
    return false;
}

void ExtractSystem::sendPair(InstanceHandle h, const WorldPose& prev, const WorldPose& curr)
{
    // Both relative to the chunk of the current pose
    const ChunkPose c = toChunkPose(curr);
    renderSink->setDynamic(h, c.chunk, dynamicPose(prev, chunkOrigin(c.chunk)), c.pose);
}

void ExtractSystem::send(const Tracked& t)
{
    if (t.handle && t.hasPose) sendPair(t.handle, t.prev, t.curr);
}

void ExtractSystem::syncInstance(Tracked& t)
{
    if (!renderSink) return;
    MeshId mesh                = 0;
    const RenderInstance* inst = w.registry().tryRead<RenderInstance>(t.entity);
    if (inst && !inst->mesh.empty()) mesh = inst->cachedMesh ? inst->cachedMesh : w.findMesh(inst->mesh);
    const bool drawn = t.hasPose && mesh != 0;
    if (t.handle && (!drawn || t.mesh != mesh))
    {
        renderSink->removeDynamic(t.handle);
        t.handle = InstanceHandle();
        t.mesh   = 0;
    }
    if (drawn && !t.handle)
    {
        // On the retry list until the instance exists and has its creation snap (addDynamic or setDynamic may throw)
        if (!t.syncPending)
        {
            t.syncPending = true;
            unposed.push_back(t.entity);
        }
        t.handle = renderSink->addDynamic(mesh);
        t.mesh   = mesh;
        // A creation always sends the current pose as a snap
        t.prev        = t.curr;
        t.snap        = true;
        t.posePending = true;
    }
    // The creation snap (again after a setDynamic that threw: a re-delivered record or the retry list)
    if (t.handle && t.posePending)
    {
        send(t);
        t.posePending = false;
    }
    if (t.syncPending)
    {
        t.syncPending = false;
        // Done: off the retry list (pushed last by the creation above; an older entry is skipped by the retry)
        if (!unposed.empty() && unposed.back() == t.entity) unposed.pop_back();
    }
}

void ExtractSystem::retryPending()
{
    // Removals the sink refused first (their entities are gone)
    while (!removals.empty())
    {
        renderSink->removeDynamic(removals.back());
        removals.pop_back();
    }
    std::vector<Entity> list;
    list.swap(unposed);
    for (size_t i = 0; i < list.size(); ++i)
    {
        Tracked* t = find(list[i]);
        if (!t || !t->syncPending || !w.registry().alive(t->entity)) continue;
        try
        {
            syncInstance(*t);
        }
        catch (...)
        {
            // The rest stays for the next run (the failed one is still pending)
            unposed.insert(unposed.end(), list.begin() + std::ptrdiff_t(i), list.end());
            throw;
        }
    }
}

void ExtractSystem::track(Entity e)
{
    if (Tracked* t = find(e))
    {
        // Delivered again (a sink that threw), or a mesh registered since: the instance follows the component
        syncInstance(*t);
        return;
    }
    if (slotOf.size() <= e.index) slotOf.resize(size_t(e.index) + 1, -1);
    Tracked t;
    t.entity = e;
    WorldPose p;
    Source s;
    t.hasPose = sourcePose(e, s, p);
    t.source  = s;
    if (t.hasPose)
    {
        t.prev = t.curr = t.seen = p;
        t.snaps                  = ++snapSeq;
    }
    t.snap         = true;
    t.processedRun = runId;
    // A pending Transform teleport is this creation snap: not again in the next pair (in a step: like the extract's
    // own teleport snap; between the steps: like one the catch-up sent)
    if (s == Source::Transform && w.registry().read<Transform>(e).wasTeleported())
    {
        if (w.inStep())
            t.snapStep = w.stepIndex();
        else
        {
            t.teleportHandled  = true;
            t.handledTeleports = w.registry().read<Transform>(e).teleportCount();
        }
    }
    tracked.push_back(t);
    slotOf[e.index] = int32_t(tracked.size() - 1);
    if (s != Source::Transform) queueRemote(tracked.back());
    syncInstance(tracked.back());
}

void ExtractSystem::untrack(Entity e)
{
    Tracked* t = find(e);
    if (!t) return;
    const InstanceHandle h = t->handle;
    // The entry goes first (a throwing sink must not leave a tracked entry of a dead entity)
    const size_t i  = size_t(slotOf[e.index]);
    slotOf[e.index] = -1;
    if (i + 1 != tracked.size())
    {
        tracked[i]                      = tracked.back();
        slotOf[tracked[i].entity.index] = int32_t(i);
    }
    tracked.pop_back();
    if (h)
    {
        // Kept until the sink removed it (a throwing sink: retried by the next extract or catch-up)
        removals.push_back(h);
        renderSink->removeDynamic(h);
        removals.pop_back();
    }
}

void ExtractSystem::queueRemote(Tracked& t)
{
    // Membership by entity (not on the Tracked entry): an entity untracked and tracked again (a name record) between
    // two commits keeps its one entry
    if (inRemote.insert(entityKey(t.entity)).second) remote.push_back(t.entity);
}

void ExtractSystem::consumeLog()
{
    if (consumer < 0) return;
    const Registry& reg = w.registry();
    w.structuralLog().consume(
        consumer,
        [&](const StructuralLog::Record& r)
        {
            if (r.component == RenderInstance::id)
            {
                // A destroy, a remove, or the remove of a name record (remove + add)
                if (!r.added)
                    untrack(r.entity);
                else if (reg.alive(r.entity) && reg.has<RenderInstance>(r.entity))
                    track(r.entity);
            }
            else if (r.component == LightSource::id)
            {
                // S36b: a light is added and removed with its component
                if (!r.added)
                    untrackLight(r.entity);
                else if (reg.alive(r.entity) && reg.has<LightSource>(r.entity))
                    trackLight(r.entity);
            }
            else if (r.component == RenderPose::id || r.component == Parent::id || r.component == NetOwner::id)
                sourcesDirty = true;
        });
    if (!sourcesDirty) return;
    // A RenderPose, Parent or owner changed: the sources are evaluated again (switches snap when processed). The flag
    // is a member: a sink that throws for a later record keeps it set for the re-delivery
    for (Tracked& t : tracked)
    {
        Entity owner;
        const Source s = sourceOf(t.entity, owner);
        if (s != Source::Transform || t.source != Source::Transform) queueRemote(t);
    }
    sourcesDirty = false;
}

void ExtractSystem::processStep(Tracked& t, uint64_t step)
{
    t.processedRun = runId;
    WorldPose pose;
    Source s;
    if (!sourcePose(t.entity, s, pose))
    {
        // Pending (a remote entity without its RenderPose): no instance until it has one
        t.source = Source::Pending;
        if (t.hasPose)
        {
            t.hasPose = false;
            syncInstance(t);
        }
        return;
    }
    if (!t.hasPose)
    {
        // Its first pose: a snap
        t.hasPose = true;
        t.source  = s;
        t.prev = t.curr = t.seen = pose;
        t.snap                   = true;
        t.snaps                  = ++snapSeq;
        syncInstance(t);
        return;
    }
    const Registry& reg = w.registry();
    // The source switched (RenderPose added or removed), or the space changed (S21: a teleport pair in the new space)
    bool snap         = s != t.source || pose.space != t.curr.space;
    uint64_t snapStep = t.snapStep;
    if (s == Source::Transform)
    {
        const Transform& tr = reg.read<Transform>(t.entity);
        if (tr.wasTeleported())
        {
            // A teleport of this step (or between the steps; one the catch-up already sent stays a snap only there,
            // a later one snaps again: the teleport count differs)
            snap     = snap || !t.teleportHandled || tr.teleportCount() != t.handledTeleports;
            snapStep = step;
        }
        // A late teleport (phases 7-8 of the last step, after its extract)
        else if (tr.wasTeleportedLastStep() && t.snapStep + 1 != step)
            snap = true;
    }
    else
    {
        Entity owner;
        sourceOf(t.entity, owner);
        if (owner.valid() && (reg.read<RenderPose>(owner).flags & RenderPose::kTeleported)) snap = true;
    }
    const WorldPose prev = snap ? pose : t.curr;
    // Idle: the pair is already (curr, curr) at this pose (a teleport the catch-up handled is done with)
    if (!snap && samePose(pose, t.curr) && samePose(t.prev, t.curr))
    {
        // The step's teleport (one the catch-up sent) is handled: a late-teleport check in the next step does not snap
        // again
        t.teleportHandled = false;
        t.snapStep        = snapStep;
        return;
    }
    staged.push_back(Staged{t.entity, prev, pose, snap, s, snapStep});
    if (t.handle) sendPair(t.handle, prev, pose);
}

void ExtractSystem::run()
{
    if (!renderSink) return;
    runId++;
    staged.clear();
    retryPending();
    // Instances are created and freed from the structural log first (creations snap at the current pose)
    consumeLog();
    const uint64_t step = w.stepIndex();
    // The entities whose Transform changed in this or the previous step (phase 6's list)
    for (uint32_t index : w.changedTransforms())
    {
        if (index >= slotOf.size() || slotOf[index] < 0) continue;
        Tracked& t = tracked[size_t(slotOf[index])];
        if (t.processedRun != runId && w.registry().alive(t.entity)) processStep(t, step);
    }
    // RenderPose sources (their flags change in phase 8), composed children and pending remote entities
    for (size_t i = 0; i < remote.size(); ++i)
    {
        Tracked* t = find(remote[i]);
        if (t && t->processedRun != runId) processStep(*t, step);
    }
    // Unsettled pairs and the last step's late movers (their pair settles to (curr, curr) also when they are no longer
    // in phase 6's list)
    for (Entity e : settle)
    {
        Tracked* t = find(e);
        if (t && t->processedRun != runId && w.registry().alive(e)) processStep(*t, step);
    }
}

void ExtractSystem::commit()
{
    if (!renderSink) return;
    settle.clear();
    for (const Staged& s : staged)
    {
        Tracked* t = find(s.entity);
        if (!t) continue;
        t->prev            = s.prev;
        t->curr            = s.curr;
        t->seen            = s.curr;
        t->snap            = s.snap;
        t->source          = s.source;
        t->snapStep        = s.snapStep;
        t->teleportHandled = false;
        if (s.snap) t->snaps = ++snapSeq;
        // Not yet (curr, curr): the next step's extract settles it
        if (!samePose(s.prev, s.curr)) settle.push_back(s.entity);
    }
    staged.clear();
    // Moved in phases 7-8 (after the phase-6 list): their pair follows in the next step; the catch-up baseline is
    // their pose now, so a catch-up before that step does not send them as edits (a paused world's catch-up sends
    // their pair, catchUpLateMovers)
    for (Entity e : w.changedDuringLastStep())
    {
        Tracked* t = find(e);
        if (!t || t->processedRun == runId || !t->hasPose) continue;
        WorldPose p;
        Source s;
        if (sourcePose(e, s, p)) t->seen = p;
        settle.push_back(e);
    }
    // The remote list keeps the tracked entities whose source is not Transform (once each)
    size_t kept = 0;
    for (size_t i = 0; i < remote.size(); ++i)
    {
        Tracked* t = find(remote[i]);
        Entity owner;
        if (!t || (t->source == Source::Transform && sourceOf(t->entity, owner) == Source::Transform))
        {
            inRemote.erase(entityKey(remote[i]));
            continue;
        }
        remote[kept++] = remote[i];
    }
    remote.resize(kept);
    // S36b: the lights at the step's pose (their pair shifts)
    syncLights(true);
    committedSteps++;
    // The step covered the edits made before it
    dirty = false;
}

void ExtractSystem::abort()
{
    // The pairs sent by the aborted extract are recomputed from the committed state by the rerun (the same pairs)
    staged.clear();
}

bool ExtractSystem::catchUp()
{
    if (!renderSink) return false;
    // Steps ran since the last call: their extract sent the changes (edits before a step are in its pair; an edit
    // after the last step, e.g. a tool call after the step tool, keeps dirty set)
    const bool stepped     = committedSteps != committedAtLastCatchUp;
    committedAtLastCatchUp = committedSteps;
    // An instance creation (or its snap) or a removal that the sink refused is retried in any frame (not an edit)
    if (!unposed.empty() || !removals.empty()) retryPending();
    if (!dirty && (stepped || !(w.isPaused() || w.timeScale() == 0 || w.structuralLog().unread(consumer) > 0)))
        return false;
    dirty = false;
    runId++;
    try
    {
        catchUpRun();
    }
    catch (...)
    {
        // A sink that threw: the edits are still unsent (pairs and baselines commit after the send); the next catch-up
        // runs again also in a running world without a new edit
        dirty = true;
        throw;
    }
    return true;
}

void ExtractSystem::catchUpRun()
{
    consumeLog();
    const Registry& reg = w.registry();
    for (Tracked& t : tracked)
    {
        if (!reg.alive(t.entity)) continue;
        WorldPose pose;
        Source s;
        if (!sourcePose(t.entity, s, pose))
        {
            t.source = Source::Pending;
            if (t.hasPose)
            {
                t.hasPose = false;
                syncInstance(t);
            }
            continue;
        }
        if (!t.hasPose)
        {
            t.hasPose = true;
            t.source  = s;
            t.prev = t.curr = t.seen = pose;
            t.snap                   = true;
            t.snaps                  = ++snapSeq;
            syncInstance(t);
            continue;
        }
        // A pending teleport is handled once (a teleport to the same pose still ends the pair in flight; a later
        // teleport counts again)
        const Transform& tr = reg.read<Transform>(t.entity);
        const bool teleport = s == Source::Transform && tr.wasTeleported() &&
                              (!t.teleportHandled || tr.teleportCount() != t.handledTeleports);
        if (s == t.source && !teleport && samePose(pose, t.seen)) continue;
        // Sent first: a sink that throws leaves the pair and the baseline as they were (the next catch-up retries)
        if (t.handle) sendPair(t.handle, pose, pose);
        if (teleport)
        {
            t.teleportHandled  = true;
            t.handledTeleports = tr.teleportCount();
        }
        t.source = s;
        t.prev = t.curr = t.seen = pose;
        t.snap                   = true;
        t.snaps                  = ++snapSeq;
    }
    // No step follows (paused, time scale 0): the last step's phase 7-8 movers get the pair that step would send (from
    // the committed curr, interpolated at the frozen alpha like every other mover); the next step settles it
    if (w.isPaused() || w.timeScale() == 0)
    {
        for (Entity e : settle)
        {
            Tracked* t = find(e);
            if (!t || !t->hasPose || !reg.alive(e) || t->source != Source::Transform) continue;
            WorldPose pose;
            Source s;
            if (!sourcePose(e, s, pose) || s != Source::Transform || samePose(pose, t->curr)) continue;
            // A late teleport snaps here (and is handled for the next step's late-teleport check)
            const uint64_t lastStep = w.stepIndex() - 1;
            const bool snap         = (reg.read<Transform>(e).wasTeleportedLastStep() && t->snapStep != lastStep) ||
                              pose.space != t->curr.space;
            const WorldPose prev = snap ? pose : t->curr;
            // Sent first (a throwing sink changes nothing)
            if (t->handle) sendPair(t->handle, prev, pose);
            t->prev = prev;
            t->curr = t->seen = pose;
            t->snap           = snap;
            if (snap)
            {
                t->snapStep = lastStep;
                t->snaps    = ++snapSeq;
            }
        }
    }
    // S36b: the lights at their current pose (edits between steps snap)
    syncLights(false);
}

//==================================================================================================
// Local lights (S36b)

ExtractSystem::TrackedLight* ExtractSystem::findLight(Entity e)
{
    if (e.index >= lightSlot.size() || lightSlot[e.index] < 0) return nullptr;
    TrackedLight& l = lights[size_t(lightSlot[e.index])];
    return l.entity == e ? &l : nullptr;
}

const ExtractSystem::TrackedLight* ExtractSystem::findLight(Entity e) const
{
    return const_cast<ExtractSystem*>(this)->findLight(e);
}

void ExtractSystem::trackLight(Entity e)
{
    TrackedLight* existing = findLight(e);
    if (!existing)
    {
        if (lightSlot.size() <= e.index) lightSlot.resize(size_t(e.index) + 1, -1);
        TrackedLight l;
        l.entity = e;
        lights.push_back(l);
        lightSlot[e.index] = int32_t(lights.size() - 1);
        existing           = &lights.back();
    }
    TrackedLight& l = *existing;
    l.source        = w.registry().read<LightSource>(e);
    WorldPose p;
    Source s;
    l.hasPose = w.registry().has<Transform>(e) && sourcePose(e, s, p);
    if (l.hasPose) l.prev = l.curr = p;
    sendLight(l, 1.0);
}

void ExtractSystem::untrackLight(Entity e)
{
    TrackedLight* l = findLight(e);
    if (!l) return;
    const LightHandle h = l->handle;
    const size_t i      = size_t(lightSlot[e.index]);
    lightSlot[e.index]  = -1;
    if (i + 1 != lights.size())
    {
        lights[i]                         = lights.back();
        lightSlot[lights[i].entity.index] = int32_t(i);
    }
    lights.pop_back();
    if (h) renderSink->removeLight(h);
}

void ExtractSystem::sendLight(TrackedLight& l, double alpha)
{
    if (!renderSink) return;
    const LightSource& src = l.source;
    std::string invalid    = src.invalidReason();
    if (!invalid.empty() && !l.invalidLogged)
    {
        l.invalidLogged = true;
        SAIGA_LOG_E("extract", l.entity, "has an invalid LightSource (%s); it is not rendered", invalid.c_str());
    }
    if (invalid.empty()) l.invalidLogged = false;
    const bool lit = l.hasPose && src.on && invalid.empty();
    if (!lit)
    {
        if (l.handle) renderSink->removeLight(l.handle);
        l.handle = LightHandle();
        return;
    }
    // The pose at alpha (S18's interpolatePose in double, as interpolatedPose)
    DynamicPose a, b;
    a.position          = l.prev.position;
    a.rotation          = l.prev.rotation;
    a.scale             = l.prev.scale;
    b.position          = l.curr.position;
    b.rotation          = l.curr.rotation;
    b.scale             = l.curr.scale;
    const DynamicPose r = alpha >= 1 ? b : Vulkan::interpolatePose(a, b, alpha);
    WorldPose at;
    at.space          = l.curr.space;
    at.position       = r.position + r.rotation * (src.offset * r.scale);
    at.rotation       = r.rotation * src.rotation;
    const ChunkPose c = toChunkPose(at);
    Vulkan::LightData d;
    d.kind              = src.kind;
    d.position          = c.pose.position.cast<float>();
    d.direction         = (at.rotation * Vec3(0, 0, -1)).normalized().cast<float>();
    d.color             = src.color.cast<float>();
    d.lumen             = float(src.lumen);
    d.range             = float(src.range);
    const double toRad  = 3.14159265358979323846 / 180;
    d.cosInner          = float(std::cos(src.innerAngleDeg * toRad));
    d.cosOuter          = float(std::cos(src.outerAngleDeg * toRad));
    d.castShadows       = src.castShadows;
    d.priority          = float(src.priority);
    d.flicker.amplitude = float(src.flickerAmplitude);
    d.flicker.frequency = float(src.flickerFrequency);
    // A seed that is the same on every peer and after a load (round 1, nit 21: the entity index differs between host
    // and client): the NetId of a replicated entity, else the persistent ID, else the entity index
    uint32_t stable = w.replication().netIdOf(l.entity);
    if (!stable)
        if (const Persistent* pid = w.registry().tryRead<Persistent>(l.entity))
            stable = uint32_t(pid->persistentId.value ^ (pid->persistentId.value >> 32));
    if (!stable) stable = l.entity.index;
    d.flicker.seed = stable * 2654435761u;
    if (!l.handle)
    {
        l.handle = renderSink->addLight(c.chunk, d);
        l.chunk  = c.chunk;
        l.data   = d;
        return;
    }
    if (l.chunk == c.chunk && sameLightData(l.data, d)) return;
    renderSink->setLight(l.handle, c.chunk, d);
    l.chunk = c.chunk;
    l.data  = d;
}

void ExtractSystem::syncLights(bool step)
{
    const Registry& reg = w.registry();
    for (TrackedLight& l : lights)
    {
        if (!reg.alive(l.entity) || !reg.has<LightSource>(l.entity)) continue;
        // (sendLight compares the resulting LightData: the source is copied as is; nit 22)
        l.source = reg.read<LightSource>(l.entity);
        WorldPose p;
        Source s;
        const bool had = l.hasPose;
        l.hasPose      = reg.has<Transform>(l.entity) && sourcePose(l.entity, s, p);
        if (l.hasPose)
        {
            // A step shifts the pair (a space change snaps); a catch-up (an edit between steps) snaps
            l.prev = step && had && p.space == l.curr.space ? l.curr : p;
            l.curr = p;
        }
        sendLight(l, 1.0);
    }
}

void ExtractSystem::updateLights(double alpha)
{
    for (TrackedLight& l : lights)
        if (l.hasPose && !samePose(l.prev, l.curr)) sendLight(l, alpha);
}

size_t ExtractSystem::lightCount() const
{
    size_t n = 0;
    for (const TrackedLight& l : lights) n += l.handle ? 1 : 0;
    return n;
}

std::vector<std::pair<Entity, LightHandle>> ExtractSystem::lightEntities() const
{
    std::vector<std::pair<Entity, LightHandle>> r;
    for (const TrackedLight& l : lights)
        if (l.handle) r.emplace_back(l.entity, l.handle);
    return r;
}

std::vector<ExtractSystem::LightEntityState> ExtractSystem::lightStates() const
{
    std::vector<LightEntityState> r;
    r.reserve(lights.size());
    for (const TrackedLight& l : lights)
    {
        LightEntityState s;
        s.entity                  = l.entity;
        s.handle                  = l.handle;
        s.source                  = l.source;
        s.hasPose                 = l.hasPose;
        s.pose                    = l.curr;
        const std::string invalid = l.source.invalidReason();
        if (l.handle)
            s.state = "rendered";
        else if (!invalid.empty())
            s.state = "invalid: " + invalid;
        else if (!l.hasPose)
            s.state = "no pose";
        else if (!l.source.on)
            s.state = "off";
        else
            s.state = "no render sink";
        r.push_back(std::move(s));
    }
    return r;
}

LightHandle ExtractSystem::lightOf(Entity e) const
{
    const TrackedLight* l = findLight(e);
    return l ? l->handle : LightHandle();
}

uint64_t ExtractSystem::snapSequence(Entity e) const
{
    const Tracked* t = find(e);
    return t ? t->snaps : 0;
}

bool ExtractSystem::interpolatedPose(Entity e, double alpha, WorldPose& out, bool* teleported) const
{
    const Tracked* t = find(e);
    if (!t || !t->hasPose || !w.alive(e)) return false;
    DynamicPose a, b;
    a.position          = t->prev.position;
    a.rotation          = t->prev.rotation;
    a.scale             = t->prev.scale;
    b.position          = t->curr.position;
    b.rotation          = t->curr.rotation;
    b.scale             = t->curr.scale;
    const DynamicPose r = Vulkan::interpolatePose(a, b, alpha);
    out.space           = t->curr.space;
    out.position        = r.position;
    out.rotation        = r.rotation;
    out.scale           = r.scale;
    if (teleported) *teleported = t->snap;
    return true;
}

}  // namespace Engine
}  // namespace Saiga
