/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

/**
 * Racing tech demo (samples/games/racing, docs/engine_design.md §18.4) on the engine app: drive a car around a track on
 * the terrain of the world prototype. --demo: the AI driver races. --scenarios: the game's scenarios.
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
    // The pedals and the steering follow the keys (released: neutral)
    bool wasActive   = false;
    setup.applyInput = [&wasActive](const InputMap& in, World& w, Entity player)
    {
        const bool active = !in.neutral();
        if (!player.valid() || !w.has<VehicleControls>(player) || (!active && !wasActive)) return;
        wasActive          = active;
        VehicleControls& c = w.get<VehicleControls>(player);
        c.throttle         = in.axis("throttle");
        c.steer            = in.axis("steer");
    };
    setup.renderHooks.init     = [&](EngineApp& app) { renderer.init(app); };
    setup.renderHooks.shutdown = [&](EngineApp&) { renderer.shutdown(); };
    setup.renderHooks.transfer = [&](vk::CommandBuffer cmd) { renderer.transfer(cmd); };
    setup.renderHooks.render   = [&](vk::CommandBuffer cmd) { renderer.render(cmd); };
    setup.renderHooks.gui      = [&]() { renderer.gui(); };

    return engineMain(argc, argv, setup, {"--demo"},
                      [](EngineApp& app, const AppOptions& options)
                      {
                          app.showStatistics(false);
                          World& w = app.world();
                          if (options.has("--demo"))
                          {
                              w.spawn<RaceDriver>(Vec3::Zero());
                              // Race a while before the first frame (a screenshot shows the car on the track)
                              w.step(60 * 10);
                          }
                          else
                          {
                              Player p;
                              p.characterId  = 1;
                              const Entity e = w.spawn<Car>(Vec3::Zero(), p, PlayerInput{});
                              app.setLocalPlayer(e);
                          }
                      });
}
