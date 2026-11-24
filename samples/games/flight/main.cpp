/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

/**
 * Flight simulator tech demo (samples/games/flight, docs/engine_design.md §18.4) on the engine app: fly a small
 * airplane over a 32 km generated overworld (S23c: the integer OverworldGenerator with forests, rocks, grass and ruin
 * sites). --demo: the autopilot flies a circuit past the fixed ruin. --scenarios: the game's scenarios.
 * --coop [delay]: co-op over loopback in one process (S16b): the client's plane next to the host's autopilot (with
 * --demo the client's plane is flown by the autopilot on the client too).
 *
 * Controls (data/flight/input.json): W/S pitch, A/D roll, Q/E yaw, shift/ctrl throttle, R respawn, G GUI, ESC quit.
 */

#include "saiga/engine/app/EngineMain.h"
#include "saiga/engine/components/Player.h"
#include "saiga/engine/components/PlayerInput.h"
#include "saiga/engine/lighting/EnvironmentWrites.h"

#include "games/common/Environment.h"
#include "games/flight/FlightRender.h"

#include <iostream>

using namespace Saiga;
using namespace Saiga::Engine;
using namespace Saiga::Game;

int main(int argc, char** argv)
{
    auto context = FlightContext::create();
    FlightRender renderer(context);

    EngineAppSetup setup;
    setup.name               = "flight";
    setup.render.msaaSamples = 4;
    // The data directory: data/flight (engineMain, --data)
    setup.world = flightWorldSetup(context, "");
    // S36: the time of day and the weather (data/flight/environment.json), ACES
    enableEnvironment(setup, "flight");
    setup.scenarios       = flightScenarios;
    setup.networkVariants = true;
    setup.checkOptions    = clampCoopDemoDelay;
    // --coop: the joined peer's plane (R-MECH-1); a leaving peer's driver goes, its plane stays empty
    setup.onPeerJoined = [](EngineApp& app, World& host, uint8_t peer)
    { flightPeerJoined(host, peer, app.options().has("--demo")); };
    setup.onPeerLeft = [](EngineApp&, World& host, uint8_t peer) { removeDriver(host, peer); };
    // The sticks follow the keys (released: centered); the throttle is a lever (its keys move it); the local player is
    // the driver
    bool wasActive   = false;
    setup.applyInput = [&wasActive](const InputMap& in, World& w, Entity player)
    {
        const bool active = !in.neutral();
        // The autopilot on a co-op client pauses while the player flies
        if (player.valid() && w.has<DemoPilot>(player)) w.get<DemoPilot>(player).suspended = active;
        if (!player.valid() || !w.has<AircraftControls>(player) || (!active && !wasActive)) return;
        wasActive           = active;
        AircraftControls& c = w.get<AircraftControls>(player);
        c.pitch             = in.axis("pitch");
        c.roll              = in.axis("roll");
        c.yaw               = in.axis("yaw");
        c.throttleChange    = in.axis("throttle");
    };
    setup.renderHooks.init         = [&](EngineApp& app) { renderer.init(app); };
    setup.renderHooks.shutdown     = [&](EngineApp&) { renderer.shutdown(); };
    setup.renderHooks.beforeRender = [&](EngineApp&, double alpha) { renderer.beforeRender(alpha); };
    setup.renderHooks.gui          = [&]() { renderer.gui(); };
    setup.onCameraReleased         = [&](EngineApp&) { renderer.cameraReleased(); };

    return engineMain(argc, argv, setup, {"--demo"},
                      [&wasActive](EngineApp& app, const AppOptions& options)
                      {
                          World& w = app.world();
                          // Regression (CodeRabbit, S13b): a save taken with the stick held restored the stick values,
                          // and neutral input never cleared them (the gate writes only active input and its release).
                          // The local player's sticks are device input: a load centers them (the throttle lever is
                          // state and stays); the next active input or an agent writes them again.
                          w.addLoadListener(
                              [&app, &wasActive](const LoadInfo&)
                              {
                                  wasActive      = false;
                                  World& world   = app.world();
                                  const Entity p = app.localPlayer();
                                  if (!p.valid() || !world.alive(p) || !world.has<AircraftControls>(p)) return;
                                  AircraftControls& c = world.get<AircraftControls>(p);
                                  c.pitch = c.roll = c.yaw = c.throttleChange = 0;
                              });
                          // The start state on the host (the hidden host world with --coop; the client's plane comes
                          // from the join policy)
                          World& start = app.startWorld();
                          if (options.has("--demo") || options.coop)
                              start.spawn<Autopilot>(Vec3::Zero());
                          else
                              app.setLocalPlayer(spawnDriven<Aircraft>(start, 0, 0).driver);
                          // S36: the demo's day passes in 10 minutes from 6:00 (sunrise, noon, sunset, the night with
                          // the moon; --time-of-day / --day-length override it after this)
                          if (options.has("--demo"))
                          {
                              WorldTimeChange day;
                              day.hours = 6.0;
                              day.rate  = 86400.0 / 600.0;
                              std::string error;
                              if (!setWorldTime(start, day, error))
                                  std::cout << "--demo: the day was not set: " << error << std::endl;
                          }
                          // Fly a while before the first frame, so the demo starts in a turn (and a screenshot shows
                          // it)
                          if (options.has("--demo")) app.stepAll(60 * 12);
                      });
}
