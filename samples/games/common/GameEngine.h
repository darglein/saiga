/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/components/RenderPose.h"
#include "saiga/engine/components/Transform.h"
#include "saiga/engine/gameplay/Behaviour.h"
#include "saiga/engine/gameplay/TickPolicy.h"
#include "saiga/engine/world/World.h"

#include "games/common/GameMath.h"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <functional>
#include <memory>
#include <vector>

namespace Saiga
{
namespace Game
{
using Engine::Entity;
using Engine::World;

// The tech demos' game objects tick every step wherever they are (a demo has no player near the autopilot or the AI
// driver; the distance policy of §8.4 would tick them every 0.5 s)
inline const Engine::TickPolicy* everyStepTickPolicy()
{
    static const Engine::TickPolicy policy{32, 1, 6, 0, 0};
    return &policy;
}

// The game object of an entity as T (nullptr if it has none or another class); for GUI buttons between steps that
// act on the demo objects too (they have no Player, so no player action reaches them)
template <class T>
T* gameObjectAs(World& w, Entity e)
{
    if (!w.alive(e) || !w.has<Engine::Behaviour>(e)) return nullptr;
    return dynamic_cast<T*>(w.read<Engine::Behaviour>(e).object.get());
}

// The tree count of the world prototype (SAIGA_WORLD_TREES overrides it, e.g. for slow GPUs)
inline int worldTreeCount(int fallback = 200000)
{
    const char* t = std::getenv("SAIGA_WORLD_TREES");
    return t ? std::atoi(t) : fallback;
}

// The time between two rendered frames (the first frame: one step), at most 0.1 s
class FrameClock
{
   public:
    double tick(double firstDt = 1.0 / 60.0)
    {
        auto now  = std::chrono::steady_clock::now();
        double dt = started ? std::chrono::duration<double>(now - last).count() : firstDt;
        last      = now;
        started   = true;
        return std::min(dt, 0.1);
    }

   private:
    std::chrono::steady_clock::time_point last;
    bool started = false;
};

// Whether the render side reads an entity's RenderPose instead of its Transform (S16b): a remote entity on a client
// (its Transform is the newest snapshot; RenderPose is interpolated from the buffer, §13.4). Owned and local entities,
// and every entity on the host, are drawn from Transform
inline bool readsRenderPose(const World& w, Entity e)
{
    return !w.isHost() && w.ownerOf(e) != w.localPeer() && w.has<Engine::RenderPose>(e);
}

/**
 * Render-side cache of one entity's pose over the last two steps (docs/engine_design.md §18.4, transitional until the
 * render sink of S19): record() after every step (RenderCaches calls it from a step listener), reset() after a load;
 * poses are interpolated with World::alpha(). A teleport (respawn) is not interpolated and reported once (the chase
 * camera snaps). An edit between steps (a paused world, a tool) is drawn at once: interpolated() replaces the history
 * when the Transform differs from the last recorded pose. S16b: a remote entity on a client is recorded from its
 * RenderPose (readsRenderPose; its teleport flag snaps).
 */
class PoseHistory
{
   public:
    struct Pose
    {
        Vec3 position = Vec3::Zero();
        Quat rotation = Quat::Identity();
    };

    void reset() { valid = false; }

    // The pose between the last two steps (alpha: World::alpha()); false if the entity is not alive
    bool interpolated(World& w, Entity entity, double alpha, Pose& out)
    {
        if (!w.alive(entity) || !w.has<Engine::Transform>(entity)) return false;
        if (!valid)
            record(w, entity);
        else if (!readsRenderPose(w, entity))
        {
            // Between steps the Transform is authoritative: a pose that differs from the last recorded one (an edit
            // after the last step: set_transform, set_component, a teleport while paused) replaces the history at
            // once, without interpolation; a teleport (its flag is not shifted before the next step) also snaps the
            // camera
            const Engine::Transform& t = w.read<Engine::Transform>(entity);
            // Regression (CodeRabbit, S13b): a teleport to the current pose kept interpolating from the previous
            // one; a pending teleport (not yet handled since the last step) snaps also when the pose is unchanged
            const bool pendingTeleport = t.wasTeleported() && !teleportHandled;
            if (pendingTeleport || !samePose(current, t))
            {
                previous = current = Pose{t.position(), t.rotation()};
                if (t.wasTeleported())
                {
                    teleported      = true;
                    teleportHandled = true;
                }
            }
        }
        out.position = previous.position + (current.position - previous.position) * alpha;
        out.rotation = previous.rotation.slerp(alpha, current.rotation);
        return true;
    }
    // A teleport since the last call (the camera should snap)
    bool takeTeleport()
    {
        bool t     = teleported;
        teleported = false;
        return t;
    }
    // After every step (RenderCaches); true if the step teleported the entity (the caller flushes its other caches)
    bool record(World& w, Entity entity)
    {
        if (!w.alive(entity) || !w.has<Engine::Transform>(entity))
        {
            valid = false;
            return false;
        }
        Pose now;
        bool snap = false;
        if (readsRenderPose(w, entity))
        {
            // A remote entity on a client: the interpolated pose of the step; its teleport flag is this step's snap
            const Engine::RenderPose& p = w.read<Engine::RenderPose>(entity);
            now                         = Pose{p.position, p.rotation};
            snap                        = (p.flags & Engine::RenderPose::kTeleported) != 0;
            teleportHandled             = false;
        }
        else
        {
            const Engine::Transform& t = w.read<Engine::Transform>(entity);
            now                        = Pose{t.position(), t.rotation()};
            // Step listeners run after the step's transform flags were shifted: the step's teleport (and one between
            // the steps, e.g. set_transform) is wasTeleportedLastStep() there; wasTeleported() is always false
            snap = t.wasTeleportedLastStep();
            // Between steps a teleport stays flagged until the next step: recorded here, it is handled
            teleportHandled = t.wasTeleported();
        }
        const bool first = !valid;
        if (first || snap)
        {
            previous = now;
            // Also after a teleport between steps that interpolated() already handled: the shifted flag cannot tell it
            // from a teleport of the step itself (e.g. the platformer's restart below the kill height), and a missed
            // snap glides the camera across the map while a second snap one step later is not visible
            teleported = true;
        }
        else
            previous = current;
        current = now;
        valid   = true;
        return first || snap;
    }

   private:
    static bool samePose(const Pose& p, const Engine::Transform& t)
    {
        const Vec3& x = t.position();
        const Quat& r = t.rotation();
        return p.position(0) == x(0) && p.position(1) == x(1) && p.position(2) == x(2) && p.rotation.w() == r.w() &&
               p.rotation.x() == r.x() && p.rotation.y() == r.y() && p.rotation.z() == r.z();
    }

    Pose previous, current;
    bool valid = false, teleported = false, teleportHandled = false;
};

/**
 * The render caches of every drawn entity (S16b; transitional until S19): per entity a PoseHistory and a ring of the
 * game's render state besides the pose (State: racing wheels, the platformer's motion), recorded after every step by
 * one step listener for the entities `enumerate` lists, cleared by a load. Remote entities on a client get their state
 * from snapshots without the interpolation delay of RenderPose: the ring is read WorldConfig::interpolationDelaySteps
 * steps late (and flushed on the RenderPose teleport flag), so wheels and animation match the drawn body
 * approximately (exact matching needs the sampled host step, which S16 does not expose).
 */
template <class State>
class RenderCaches
{
   public:
    struct Entry
    {
        Entity entity;
        PoseHistory pose;
        std::deque<State> ring;  // after each step, newest last
        int delay = 0;           // steps the state is read late
        // The state after the delayed step and the one before it (interpolate with alpha)
        const State& current() const
        {
            return ring[ring.size() - 1 - size_t(std::min<int>(delay, int(ring.size()) - 1))];
        }
        const State& previous() const
        {
            const size_t c = ring.size() - 1 - size_t(std::min<int>(delay, int(ring.size()) - 1));
            return ring[c > 0 ? c - 1 : 0];
        }
        // The largest pulse (Pulse, e.g. a landing speed) of the states that became current since the caller took it
        double pending = 0;
    };
    using Enumerate = std::function<void(World&, const std::function<void(Entity)>&)>;
    using Read      = std::function<State(const World&, Entity)>;
    // A one-step value of a state (the platformer's landing speed), collected into Entry::pending when the state
    // becomes current, so a pulse of a step is seen once although several steps run in one frame
    using Pulse = std::function<double(const State&)>;

    void attach(World& world, Enumerate enumerate, Read read, Pulse pulse = Pulse())
    {
        w            = &world;
        listEntities = std::move(enumerate);
        readState    = std::move(read);
        pulseOf      = std::move(pulse);
        stepListener = w->addStepListener(
            [this]()
            {
                recordAll();
                return true;
            });
        loadListener = w->addLoadListener([this](const auto&) { entries.clear(); });
    }
    void detach()
    {
        if (!w) return;
        w->removeStepListener(stepListener);
        w->removeLoadListener(loadListener);
        w = nullptr;
        entries.clear();
    }
    // The cache of an alive entity (created with its current state on first use, e.g. between steps); nullptr if dead
    Entry* entry(Entity e)
    {
        if (!w || !e.valid() || !w->alive(e)) return nullptr;
        for (auto& en : entries)
            if (en->entity == e) return en.get();
        auto en    = std::make_unique<Entry>();
        en->entity = e;
        en->delay  = delayOf(e);
        en->pose.record(*w, e);
        en->ring.push_back(readState(*w, e));
        entries.push_back(std::move(en));
        return entries.back().get();
    }
    // Replaces an entity's state ring by its state now (an edit between steps, e.g. a respawn while paused)
    void reseed(Entity e)
    {
        if (Entry* en = entry(e))
        {
            en->ring.clear();
            en->ring.push_back(readState(*w, e));
        }
    }

   private:
    int delayOf(Entity e) const
    {
        return readsRenderPose(*w, e) ? std::max(0, w->worldConfig().interpolationDelaySteps) : 0;
    }
    void recordAll()
    {
        // Dead entities go; the listed ones are recorded (created on first sight)
        for (size_t i = 0; i < entries.size();)
            if (!w->alive(entries[i]->entity))
                entries.erase(entries.begin() + std::ptrdiff_t(i));
            else
                ++i;
        listEntities(*w,
                     [this](Entity e)
                     {
                         Entry* en = nullptr;
                         for (auto& x : entries)
                             if (x->entity == e) en = x.get();
                         if (!en)
                         {
                             entry(e);
                             return;
                         }
                         en->delay       = delayOf(e);
                         const bool snap = en->pose.record(*w, e);
                         if (snap) en->ring.clear();
                         en->ring.push_back(readState(*w, e));
                         while (int(en->ring.size()) > en->delay + 2) en->ring.pop_front();
                         // A new state became current (until the ring is filled after a flush the oldest one stays)
                         const bool advanced = snap || en->delay == 0 || int(en->ring.size()) >= en->delay + 2;
                         if (advanced && pulseOf) en->pending = std::max(en->pending, pulseOf(en->current()));
                     });
    }

    World* w         = nullptr;
    int stepListener = -1, loadListener = -1;
    Enumerate listEntities;
    Read readState;
    Pulse pulseOf;
    std::vector<std::unique_ptr<Entry>> entries;
};

// The camera hold of S14 (EngineApp::cameraHeldByTool): while a tool holds the camera the game renders from the app
// camera and does not update its own; on the release edge it resets its own camera (no glide back)
struct HeldCamera
{
    // True once when the hold ended since the last call
    bool released(bool held)
    {
        const bool r = wasHeld && !held;
        wasHeld      = held;
        return r;
    }
    bool wasHeld = false;
};

}  // namespace Game
}  // namespace Saiga
