/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// ExtractSystem (§10.3; S19b). Reads: RenderInstance, Transform and its flags, RenderPose and its flags, Parent,
// LocalTransform, the structural log, the owner table. Writes: the render sink's dynamic instances (no components).

#include "saiga/engine/sync/ExtractSystem.h"

#include "saiga/engine/components/LocalTransform.h"
#include "saiga/engine/components/Parent.h"
#include "saiga/engine/components/RenderInstance.h"
#include "saiga/engine/components/RenderPose.h"
#include "saiga/engine/components/Transform.h"
#include "saiga/engine/ecs/EntityLog.h"
#include "saiga/engine/net/NetOwner.h"
#include "saiga/engine/net/Replication.h"
#include "saiga/engine/world/World.h"

#include <cmath>
#include <cstring>

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
    c.chunk.x     = int32_t(std::floor(p.position(0) / chunkSize));
    c.chunk.z     = int32_t(std::floor(p.position(2) / chunkSize));
    c.pose        = dynamicPose(p, chunkOrigin(c.chunk));
    return c;
}

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
        for (Tracked& t : tracked)
            if (t.handle) renderSink->removeDynamic(t.handle);
        tracked.clear();
        slotOf.clear();
        remote.clear();
        staged.clear();
        settle.clear();
        unposed.clear();
        detachConsumer();
    }
    renderSink = sink;
    if (!renderSink) return;
    attachConsumer();
    // The existing entities (the records before the consumer existed are not read): snaps
    std::vector<Entity> existing;
    w.registry().view<const RenderInstance>().each([&](Entity e, const RenderInstance&) { existing.push_back(e); });
    for (Entity e : existing) track(e);
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
        t.handle = renderSink->addDynamic(mesh);
        t.mesh   = mesh;
        // A creation always sends the current pose as a snap
        t.prev        = t.curr;
        t.snap        = true;
        t.posePending = true;
        unposed.push_back(t.entity);
    }
    // The creation snap (again after a setDynamic that threw: a re-delivered record or the retry list)
    if (t.handle && t.posePending)
    {
        send(t);
        t.posePending = false;
        // Sent: off the retry list (pushed last by the creation above; a re-delivered one is skipped by the retry)
        if (!unposed.empty() && unposed.back() == t.entity) unposed.pop_back();
    }
}

void ExtractSystem::retryUnposed()
{
    std::vector<Entity> list;
    list.swap(unposed);
    for (size_t i = 0; i < list.size(); ++i)
    {
        Tracked* t = find(list[i]);
        if (!t || !t->handle || !t->posePending) continue;
        try
        {
            send(*t);
        }
        catch (...)
        {
            // The rest stays for the next run
            unposed.insert(unposed.end(), list.begin() + std::ptrdiff_t(i), list.end());
            throw;
        }
        t->posePending = false;
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
            t.teleportHandled = true;
    }
    tracked.push_back(t);
    slotOf[e.index] = int32_t(tracked.size() - 1);
    if (s != Source::Transform) remote.push_back(e);
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
    if (h) renderSink->removeDynamic(h);
}

void ExtractSystem::consumeLog()
{
    if (consumer < 0) return;
    const Registry& reg = w.registry();
    bool sourcesDirty   = false;
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
            else if (r.component == RenderPose::id || r.component == Parent::id || r.component == NetOwner::id)
                sourcesDirty = true;
        });
    if (!sourcesDirty) return;
    // A RenderPose, Parent or owner changed: the sources are evaluated again (switches snap when processed)
    for (const Tracked& t : tracked)
    {
        Entity owner;
        const Source s = sourceOf(t.entity, owner);
        if (s != Source::Transform || t.source != Source::Transform) remote.push_back(t.entity);
    }
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
    bool snap           = s != t.source;  // the source switched (RenderPose added or removed)
    uint64_t snapStep   = t.snapStep;
    if (s == Source::Transform)
    {
        const Transform& tr = reg.read<Transform>(t.entity);
        if (tr.wasTeleported())
        {
            // A teleport of this step (or between the steps; one the catch-up already sent stays a snap only there)
            snap     = snap || !t.teleportHandled;
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
    retryUnposed();
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
        if (!t || t->remoteRun == runId) continue;
        Entity owner;
        if (t->source == Source::Transform && sourceOf(t->entity, owner) == Source::Transform) continue;
        t->remoteRun   = runId;
        remote[kept++] = remote[i];
    }
    remote.resize(kept);
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
    // A creation snap that did not reach the sink is sent again in any frame (it is not an edit)
    if (!unposed.empty()) retryUnposed();
    if (!dirty && (stepped || !(w.isPaused() || w.timeScale() == 0 || w.structuralLog().unread(consumer) > 0)))
        return false;
    dirty = false;
    runId++;
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
        // A pending teleport is handled once (a teleport to the same pose still ends the pair in flight)
        const bool teleport =
            s == Source::Transform && reg.read<Transform>(t.entity).wasTeleported() && !t.teleportHandled;
        if (s == t.source && !teleport && samePose(pose, t.seen)) continue;
        if (teleport) t.teleportHandled = true;
        t.source = s;
        t.prev = t.curr = t.seen = pose;
        t.snap                   = true;
        t.snaps                  = ++snapSeq;
        send(t);
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
            const bool snap         = reg.read<Transform>(e).wasTeleportedLastStep() && t->snapStep != lastStep;
            t->prev                 = snap ? pose : t->curr;
            t->curr = t->seen = pose;
            t->snap           = snap;
            if (snap)
            {
                t->snapStep = lastStep;
                t->snaps    = ++snapSeq;
            }
            send(*t);
        }
    }
    return true;
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
