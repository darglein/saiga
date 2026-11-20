/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

/**
 * Flight simulator tech demo (samples/games/flight, docs/engine_design.md §18.4) on the engine app: fly a small
 * airplane over the 8 km world prototype. --demo: the autopilot flies a circuit. --scenarios: the game's scenarios.
 *
 * Controls (data/flight/input.json): W/S pitch, A/D roll, Q/E yaw, shift/ctrl throttle, R respawn, G GUI, ESC quit.
 */

#include "saiga/engine/app/EngineMain.h"
#include "saiga/engine/components/Player.h"
#include "saiga/engine/components/PlayerInput.h"

#include "games/flight/FlightRender.h"

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
    setup.world     = flightWorldSetup(context, "");
    setup.scenarios = flightScenarios;
    // The sticks follow the keys (released: centered); the throttle is a lever (its keys move it)
    bool wasActive   = false;
    setup.applyInput = [&wasActive](const InputMap& in, World& w, Entity player)
    {
        const bool active = !in.neutral();
        if (!player.valid() || !w.has<AircraftControls>(player) || (!active && !wasActive)) return;
        wasActive           = active;
        AircraftControls& c = w.get<AircraftControls>(player);
        c.pitch             = in.axis("pitch");
        c.roll              = in.axis("roll");
        c.yaw               = in.axis("yaw");
        c.throttleChange    = in.axis("throttle");
    };
    setup.renderHooks.init     = [&](EngineApp& app) { renderer.init(app); };
    setup.renderHooks.shutdown = [&](EngineApp&) { renderer.shutdown(); };
    setup.renderHooks.transfer = [&](vk::CommandBuffer cmd) { renderer.transfer(cmd); };
    setup.renderHooks.render   = [&](vk::CommandBuffer cmd) { renderer.render(cmd); };
    setup.renderHooks.gui      = [&]() { renderer.gui(); };

    return engineMain(argc, argv, setup, {"--demo"},
                      [&wasActive](EngineApp& app, const AppOptions& options)
                      {
                          app.showStatistics(false);
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
                          if (options.has("--demo"))
                          {
                              w.spawn<Autopilot>(Vec3::Zero());
                              // Fly a while before the first frame, so the demo starts in a turn (and a screenshot
                              // shows it)
                              w.step(60 * 12);
                          }
                          else
                          {
                              Player p;
                              p.characterId  = 1;
                              const Entity e = w.spawn<Aircraft>(Vec3::Zero(), p, PlayerInput{});
                              app.setLocalPlayer(e);
                          }
                      });
}
