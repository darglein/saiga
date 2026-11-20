/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/data/DataRef.h"
#include "saiga/engine/gameplay/GameObject.h"

#include "games/common/GameEngine.h"
#include "games/platformer/CharacterController.h"
#include "games/platformer/PlatformerDemo.h"

#include <memory>

namespace Saiga
{
namespace Engine
{
class ScenarioList;
}
namespace Game
{
/**
 * The simulation of the platformer tech demo (docs/engine_design.md §18.4, S13b) in a World: the character's state in
 * components (CharacterMotion, RunState, its feet position and facing in Transform), the kinematic controller as a
 * system (characterControllerSystem: coyote time, jump buffer, variable jump height from the held jump bit, sub-stepped
 * box collision), the decisions as game objects (PlatformerHero: respawn below the level, the goal and the run time;
 * DemoRunner: the demo's input). The level is the context.
 */

struct PlatformerContext
{
    Level level;
};

enum class PlatformerComponent : uint8_t
{
    CharacterMotion = Engine::kFirstGameComponent,
    RunState,
};

// The controller's state besides the pose (R-STATE-1)
struct CharacterMotion
{
    static constexpr const char* name       = "CharacterMotion";
    static constexpr Engine::ComponentId id = Engine::ComponentId(PlatformerComponent::CharacterMotion);
    static constexpr uint32_t version       = 1;
    Engine::DataRef<CharacterParameters> model{"default"};
    Vec3 velocity          = Vec3::Zero();
    double yaw             = 0;
    bool grounded          = false;
    double airTime         = 0;
    double landingSpeed    = 0;  // set in the step that landed (m/s)
    bool jumped            = false;
    double jumpBufferTimer = 0;
    bool jumpWasHeld       = false;  // the jump bit of the previous step (a press is a new hold)
    bool jumpRequest = false;  // a press without a new hold (the demo runner holds the bit), used by the next step

    template <class V>
    void visit(V& v)
    {
        v("model", model);
        v("velocity", velocity);
        v("yaw", yaw);
        v("grounded", grounded);
        v("airTime", airTime);
        v("landingSpeed", landingSpeed);
        v("jumped", jumped);
        v("jumpBufferTimer", jumpBufferTimer);
        v("jumpWasHeld", jumpWasHeld);
        v("jumpRequest", jumpRequest);
    }
};

// The run: time since the (re)start, the time the goal was reached (-1: not yet), the demo runner's next waypoint
struct RunState
{
    static constexpr const char* name       = "RunState";
    static constexpr Engine::ComponentId id = Engine::ComponentId(PlatformerComponent::RunState);
    static constexpr uint32_t version       = 1;
    double runTime                          = 0;
    double goalTime                         = -1;
    int32_t waypoint                        = 0;

    template <class V>
    void visit(V& v)
    {
        v("runTime", runTime);
        v("goalTime", goalTime);
        v("waypoint", waypoint);
    }
};

enum class PlatformerClass : uint16_t
{
    PlatformerHero = Engine::kFirstGameClass,
    DemoRunner,
};

// Held buttons (PlayerInput::heldButtons) and actions
enum class PlatformerButton : int
{
    Jump = 0,
    Walk = 1,
};
enum class PlatformerAction : uint32_t
{
    Restart = 1,
};

// The controller of an entity's components (feet position and facing from Transform and CharacterMotion)
CharacterController characterOf(const World& w, Entity e);

class HeroBase : public Engine::GameObject
{
   public:
    void onSpawn() override;
    void update(double dt) override;
    const Engine::TickPolicy* tickPolicy() const override { return everyStepTickPolicy(); }
    // Back at the level's start
    void restart();
    // The demo restarts a few seconds after reaching the goal
    virtual bool restartsAfterTheGoal() const { return false; }
};

// The player's character (PlayerInput from the input map)
class PlatformerHero final : public HeroBase
{
   public:
    static constexpr Engine::ClassId id = Engine::ClassId(PlatformerClass::PlatformerHero);
    static constexpr const char* name   = "PlatformerHero";
};

// The demo: runs through the level by itself (writes its PlayerInput) and starts over after the goal
class DemoRunner final : public HeroBase
{
   public:
    static constexpr Engine::ClassId id = Engine::ClassId(PlatformerClass::DemoRunner);
    static constexpr const char* name   = "DemoRunner";
    void onSpawn() override;
    void update(double dt) override;
    bool restartsAfterTheGoal() const override { return true; }
};

Engine::WorldSetup platformerWorldSetup(std::shared_ptr<const PlatformerContext> context);
void platformerScenarios(Engine::ScenarioList& list);

}  // namespace Game
}  // namespace Saiga
