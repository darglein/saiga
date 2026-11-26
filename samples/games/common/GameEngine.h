/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/env.h"
#include "saiga/engine/components/RenderPose.h"
#include "saiga/engine/components/Transform.h"
#include "saiga/engine/gameplay/Behaviour.h"
#include "saiga/engine/gameplay/TickPolicy.h"
#include "saiga/engine/sync/ExtractSystem.h"
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
    const auto t = GetEnvIfSet("SAIGA_WORLD_TREES");
    return t ? std::atoi(t->c_str()) : fallback;
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

// Whether an entity is drawn from its RenderPose (S16b; S19b: the extract decides the drawn pose, this is the delay of
// the game's own render state, RenderCaches): a remote entity on a client (its Transform is the newest snapshot;
// RenderPose is interpolated from the buffer, §13.4)
inline bool readsRenderPose(const World& w, Entity e)
{
    return !w.isHost() && w.ownerOf(e) != w.localPeer() && w.has<Engine::RenderPose>(e);
}

/**
 * The render caches of the game's state besides the pose (S16b; S19b: the pose comes from the engine's extract,
 * ExtractSystem::interpolatedPose): per entity a ring of the game's render state (State: racing wheels, the
 * platformer's motion), recorded after every step by one step listener for the entities `enumerate` lists, cleared by
 * a load. Remote entities on a client get their state from snapshots without the interpolation delay of RenderPose: the
 * ring is read WorldConfig::interpolationDelaySteps steps late (and flushed on the RenderPose teleport flag), so wheels
 * and animation match the drawn body approximately (exact matching needs the sampled host step, which S16 does not
 * expose).
 */
template <class State>
class RenderCaches
{
   public:
    struct Entry
    {
        Entity entity;
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
        // The last snap of the entity the renderer saw (ExtractSystem::snapSequence: react once per snap)
        uint64_t snapsSeen = 0;
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
        en->ring.push_back(readState(*w, e));
        // The first state's pulse is collected too (an entity first seen on a landing step)
        if (pulseOf) en->pending = pulseOf(en->current());
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
    // The step teleported the entity (a remote one: its RenderPose snapped; step listeners run after the flags were
    // shifted, so a local one's teleport of the step is wasTeleportedLastStep)
    bool teleported(Entity e) const
    {
        if (readsRenderPose(*w, e))
            return (w->read<Engine::RenderPose>(e).flags & Engine::RenderPose::kTeleported) != 0;
        return w->read<Engine::Transform>(e).wasTeleportedLastStep();
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
                         const bool snap = teleported(e);
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

}  // namespace Game
}  // namespace Saiga
