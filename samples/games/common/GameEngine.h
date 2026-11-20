/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/components/Transform.h"
#include "saiga/engine/gameplay/TickPolicy.h"
#include "saiga/engine/world/World.h"

#include "games/common/GameMath.h"

#include <chrono>
#include <cstdlib>

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

/**
 * Render-side cache of one entity's pose over the last two steps (docs/engine_design.md §18.4, transitional until the
 * render sink of S19): a step listener records the Transform after every step, a load listener clears it; poses are
 * interpolated with World::alpha(). A teleport (respawn) is not interpolated and reported once (the chase camera
 * snaps).
 */
class PoseHistory
{
   public:
    struct Pose
    {
        Vec3 position = Vec3::Zero();
        Quat rotation = Quat::Identity();
    };

    void attach(World& world)
    {
        w            = &world;
        stepListener = w->addStepListener(
            [this]()
            {
                record();
                return true;
            });
        loadListener = w->addLoadListener([this](const auto&) { valid = false; });
    }
    // Removes the listeners (before this object goes away while the world lives on)
    void detach()
    {
        if (!w) return;
        w->removeStepListener(stepListener);
        w->removeLoadListener(loadListener);
        w = nullptr;
    }
    // The entity to follow (a change starts a new history)
    void follow(Entity e)
    {
        if (e == entity) return;
        entity = e;
        valid  = false;
    }
    Entity followed() const { return entity; }

    // The pose between the last two steps (alpha: World::alpha()); false if the entity is not alive
    bool interpolated(double alpha, Pose& out)
    {
        if (!w || !w->alive(entity) || !w->has<Engine::Transform>(entity)) return false;
        if (!valid) record();
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

   private:
    void record()
    {
        if (!w->alive(entity) || !w->has<Engine::Transform>(entity))
        {
            valid = false;
            return;
        }
        const Engine::Transform& t = w->read<Engine::Transform>(entity);
        Pose now{t.position(), t.rotation()};
        // Step listeners run after the step's transform flags were shifted: the step's teleport (and one between the
        // steps, e.g. set_transform) is wasTeleportedLastStep() there; wasTeleported() is always false
        if (!valid || t.wasTeleportedLastStep())
        {
            previous   = now;
            teleported = true;
        }
        else
            previous = current;
        current = now;
        valid   = true;
    }

    World* w         = nullptr;
    int stepListener = -1, loadListener = -1;
    Entity entity;
    Pose previous, current;
    bool valid = false, teleported = false;
};

}  // namespace Game
}  // namespace Saiga
