/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

/**
 * Racing tech demo (samples/games/racing, docs/engine_design.md §18.4) on the engine app: drive a car around a track on
 * the terrain of the world prototype. --demo: the AI driver races. --scenarios: the game's scenarios. --coop [delay]:
 * co-op over loopback in one process (S16b): the client's car next to the host's AI car (with --demo the client's
 * car is driven by the AI on the client too).
 *
 * Controls (data/racing/input.json): W throttle, S brake / reverse, A/D steer, R back to the track, G GUI, ESC quit.
 */

#include "saiga/engine/app/EngineMain.h"
#include "saiga/engine/components/Player.h"
#include "saiga/engine/components/PlayerInput.h"

#include "games/racing/RacingRender.h"

using namespace Saiga;
using namespace Saiga::Engine;
using namespace Saiga::Game;

int main(int argc, char** argv)
{
    auto context = RacingContext::create();
    RacingRender renderer(context);

    EngineAppSetup setup;
    setup.name               = "racing";
    setup.render.msaaSamples = 4;
    setup.world              = racingWorldSetup(context);
    setup.scenarios          = racingScenarios;
    setup.networkVariants    = true;
    setup.checkOptions       = clampCoopDemoDelay;
    // --coop: the joined peer's car (R-MECH-1); a leaving peer's driver goes, its car stays empty
    setup.onPeerJoined = [](EngineApp& app, World& host, uint8_t peer)
    { racingPeerJoined(host, peer, app.options().has("--demo")); };
    setup.onPeerLeft = [](EngineApp&, World& host, uint8_t peer) { removeDriver(host, peer); };
    // The pedals and the steering follow the keys (released: neutral); the local player is the driver
    bool wasActive   = false;
    setup.applyInput = [&wasActive](const InputMap& in, World& w, Entity player)
    {
        const bool active = !in.neutral();
        // The demo's AI on a co-op client pauses while the player drives
        if (player.valid() && w.has<DemoPilot>(player)) w.get<DemoPilot>(player).suspended = active;
        if (!player.valid() || !w.has<VehicleControls>(player) || (!active && !wasActive)) return;
        wasActive          = active;
        VehicleControls& c = w.get<VehicleControls>(player);
        c.throttle         = in.axis("throttle");
        c.steer            = in.axis("steer");
    };
    setup.renderHooks.init         = [&](EngineApp& app) { renderer.init(app); };
    setup.renderHooks.shutdown     = [&](EngineApp&) { renderer.shutdown(); };
    setup.renderHooks.beforeRender = [&](EngineApp&, double alpha) { renderer.beforeRender(alpha); };
    setup.renderHooks.gui          = [&]() { renderer.gui(); };
    setup.onCameraReleased         = [&](EngineApp&) { renderer.cameraReleased(); };

    return engineMain(argc, argv, setup, {"--demo"},
                      [&wasActive](EngineApp& app, const AppOptions& options)
                      {
                          app.showStatistics(true);  // the frame times (the World Statistics window points here)
                          World& w = app.world();
                          // Regression (CodeRabbit, S13b): a save taken with a pedal or the steering held restored
                          // them, and neutral input never cleared them. The local player's pedals and steering are
                          // device input: a load sets them neutral; the next active input or an agent writes them.
                          w.addLoadListener(
                              [&app, &wasActive](const LoadInfo&)
                              {
                                  wasActive      = false;
                                  World& world   = app.world();
                                  const Entity p = app.localPlayer();
                                  if (!p.valid() || !world.alive(p) || !world.has<VehicleControls>(p)) return;
                                  world.get<VehicleControls>(p) = VehicleControls();
                              });
                          // The start state on the host (the hidden host world with --coop; the client's car comes
                          // from the join policy)
                          World& start = app.startWorld();
                          if (options.has("--demo") || options.coop)
                              start.spawn<RaceDriver>(Vec3::Zero());
                          else
                              app.setLocalPlayer(spawnDriven<Car>(start, 0, 0).driver);
                          // Race a while before the first frame (a screenshot shows the car on the track)
                          if (options.has("--demo")) app.stepAll(60 * 10);
                      });
}
