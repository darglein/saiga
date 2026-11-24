/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

/**
 * Platformer tech demo (samples/games/platformer, docs/engine_design.md §18.4) on the engine app: one level and an
 * enclosed cellar behind a door (S21), a procedurally animated character. --demo: the character runs through the level
 * by itself; --demo --cellar: through the cellar first (the screenshot shows the cellar). --scenarios: the game's
 * scenarios. --coop [delay]: co-op over loopback in one process (S16b): the client's hero next to the host's demo
 * runner (with --demo the client's hero is run by the demo runner on the client too).
 *
 * Controls (data/platformer/input.json): WASD move (relative to the camera), shift walk, space jump, F use a door, Q/E
 * turn the camera, R restart, G GUI, ESC quit.
 */

#include "saiga/engine/app/EngineMain.h"
#include "saiga/engine/components/Player.h"
#include "saiga/engine/components/PlayerInput.h"

#include "games/common/Environment.h"
#include "games/platformer/PlatformerRender.h"

using namespace Saiga;
using namespace Saiga::Engine;
using namespace Saiga::Game;

// --demo --cellar (S21): the pre-simulation that puts the capture of frame 60 in the middle of the hero's stay in the
// cellar (entry + stay / 2 - 60; the entry step and the stay are logged by the two cellar scenarios)
constexpr int kSoloCellarSteps = 108;  // entry 33, stay 270 (Plain)
constexpr int kCoopCellarSteps = 142;  // entry 67, stay 270 (HostClientDelay, the client's hero)

int main(int argc, char** argv)
{
    auto context = std::make_shared<PlatformerContext>(PlatformerContext::demo());
    PlatformerRender renderer(context);

    EngineAppSetup setup;
    setup.name               = "platformer";
    setup.render.msaaSamples = 4;
    setup.render.clearColor  = vec4(0.62f, 0.74f, 0.86f, 1);
    setup.world              = platformerWorldSetup(context);
    // S36: the environment with the code defaults (a frozen noon outdoors; the cellar is lit by its template), ACES
    enableEnvironment(setup, "platformer");
    setup.scenarios       = platformerScenarios;
    setup.networkVariants = true;
    setup.checkOptions    = clampCoopDemoDelay;
    // --coop: the joined peer's hero (R-MECH-1); a leaving peer's hero is destroyed
    setup.onPeerJoined = [](EngineApp& app, World& host, uint8_t peer)
    {
        platformerPeerJoined(host, peer, app.options().has("--demo"),
                             app.options().has("--cellar") ? PlatformerRoute::Cellar : PlatformerRoute::Outdoor);
    };
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

    return engineMain(
        argc, argv, setup, {"--demo", "--cellar"},
        [](EngineApp& app, const AppOptions& options)
        {
            // The start state on the host (the hidden host world with --coop; the client's hero comes
            // from the join policy): the doors of the cellar (S21), then the heroes
            World& start      = app.startWorld();
            const bool cellar = options.has("--cellar");
            spawnDoors(start);
            if (options.has("--demo") || options.coop)
                spawnDemoRunner(start, cellar ? PlatformerRoute::Cellar : PlatformerRoute::Outdoor);
            else
                app.setLocalPlayer(spawnHero(start, 0, 0));
            // Run a while before the first frame (a screenshot then shows the character running over the
            // bridge; with --cellar in the middle of the stay in the cellar: N = entry + stay / 2 - 60
            // steps before the capture of frame 60, from demo_runner_visits_the_cellar (solo) and
            // pilot_visits_the_cellar in HostClientDelay (co-op, the client's hero); S21)
            if (options.has("--demo"))
                app.stepAll(cellar ? (options.coop ? kCoopCellarSteps : kSoloCellarSteps) : int(60 * 4.6));
        });
}
