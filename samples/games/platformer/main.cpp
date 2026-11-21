/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

/**
 * Platformer tech demo (samples/games/platformer, docs/engine_design.md §18.4) on the engine app: one level, a
 * procedurally animated character. --demo: the character runs through the level by itself. --scenarios: the game's
 * scenarios. --coop [delay]: co-op over loopback in one process (S16b): the client's hero next to the host's demo
 * runner (with --demo the client's hero is run by the demo runner on the client too).
 *
 * Controls (data/platformer/input.json): WASD move (relative to the camera), shift walk, space jump, Q/E turn the
 * camera, R restart, G GUI, ESC quit.
 */

#include "saiga/engine/app/EngineMain.h"
#include "saiga/engine/components/Player.h"
#include "saiga/engine/components/PlayerInput.h"

#include "games/platformer/PlatformerRender.h"

using namespace Saiga;
using namespace Saiga::Engine;
using namespace Saiga::Game;

int main(int argc, char** argv)
{
    auto context   = std::make_shared<PlatformerContext>();
    context->level = Level::demoLevel();
    PlatformerRender renderer(context);

    EngineAppSetup setup;
    setup.name               = "platformer";
    setup.render.msaaSamples = 4;
    setup.render.clearColor  = vec4(0.62f, 0.74f, 0.86f, 1);
    setup.world              = platformerWorldSetup(context);
    setup.scenarios          = platformerScenarios;
    setup.networkVariants    = true;
    setup.checkOptions       = clampCoopDemoDelay;
    // --coop: the joined peer's hero (R-MECH-1); a leaving peer's hero is destroyed
    setup.onPeerJoined = [](EngineApp& app, World& host, uint8_t peer)
    { platformerPeerJoined(host, peer, app.options().has("--demo")); };
    setup.onPeerLeft = [](EngineApp&, World& host, uint8_t peer) { removeDriver(host, peer); };
    // The demo runner on a co-op client pauses while the player moves the hero
    setup.applyInput = [](const InputMap& in, World& w, Entity player)
    {
        if (player.valid() && w.has<DemoPilot>(player)) w.get<DemoPilot>(player).suspended = !in.neutral();
    };
    setup.renderHooks.init         = [&](EngineApp& app) { renderer.init(app); };
    setup.renderHooks.shutdown     = [&](EngineApp&) { renderer.shutdown(); };
    setup.renderHooks.beforeRender = [&](EngineApp&, double alpha) { renderer.beforeRender(alpha); };
    setup.renderHooks.gui          = [&]() { renderer.gui(); };
    setup.onCameraReleased         = [&](EngineApp&) { renderer.cameraReleased(); };

    return engineMain(argc, argv, setup, {"--demo"},
                      [](EngineApp& app, const AppOptions& options)
                      {
                          // The start state on the host (the hidden host world with --coop; the client's hero comes
                          // from the join policy)
                          World& start = app.startWorld();
                          if (options.has("--demo") || options.coop)
                              start.spawn<DemoRunner>(Vec3::Zero());
                          else
                              app.setLocalPlayer(spawnHero(start, 0, 0));
                          // Run a while before the first frame (a screenshot then shows the character running over the
                          // bridge)
                          if (options.has("--demo")) app.stepAll(int(60 * 4.6));
                      });
}
