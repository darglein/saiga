/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

/**
 * Flight simulator tech demo (samples/games/flight, docs/engine_design.md §18.4): fly a small airplane over the 8 km
 * world prototype. --demo: the autopilot flies a circuit.
 *
 * Controls: W/S pitch, A/D roll, Q/E yaw, shift/ctrl throttle up/down, R respawn, G GUI, ESC quit.
 */

#include "games/common/GameInput.h"
#include "games/common/GameSampleHost.h"
#include "games/flight/FlightGame.h"

using namespace Saiga;
using namespace Saiga::Game;

class FlightSample : public GameSampleHost<FlightGame>
{
   public:
    explicit FlightSample(bool demo) { game.init(*renderer, demo, worldTreeCount()); }

   private:
    void stepGame(double dt, bool keyboardFree) override
    {
        // Regression: while ImGui had the keyboard the last stick inputs stayed applied (the throttle is a lever and
        // keeps its value)
        controls.pitch = controls.roll = controls.yaw = 0;
        if (keyboardFree)
        {
            throttle          = clampd(throttle + keyAxis(GLFW_KEY_LEFT_CONTROL, GLFW_KEY_LEFT_SHIFT) * 0.5 * dt, 0, 1);
            controls.throttle = throttle;
            controls.pitch    = keyAxis(GLFW_KEY_S, GLFW_KEY_W);
            controls.roll     = keyAxis(GLFW_KEY_A, GLFW_KEY_D);
            controls.yaw      = keyAxis(GLFW_KEY_Q, GLFW_KEY_E);
        }
        game.step(controls, dt);
    }

    AircraftControls controls;
    double throttle = 0.6;
};

int main(int argc, char** argv)
{
    initSaigaSample();
    FlightSample sample(FlightSample::hasFlag(argc, argv, "--demo"));
    sample.runGame();
    return 0;
}
