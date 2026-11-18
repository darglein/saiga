/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

/**
 * Racing tech demo (samples/games/racing, docs/engine_design.md §18.4): drive a car around a track on the terrain of the
 * world prototype. --demo: the AI driver races.
 *
 * Controls: W throttle, S brake / reverse, A/D steer, R back to the track, G GUI, ESC quit.
 */

#include "games/common/GameInput.h"
#include "games/common/GameSampleHost.h"
#include "games/racing/RacingGame.h"

using namespace Saiga;
using namespace Saiga::Game;

class RacingSample : public GameSampleHost<RacingGame>
{
   public:
    explicit RacingSample(bool demo) { game.init(*renderer, demo, worldTreeCount()); }

   private:
    void stepGame(double dt, bool keyboardFree) override
    {
        VehicleControls controls;
        if (keyboardFree)
        {
            controls.throttle = keyAxis(GLFW_KEY_S, GLFW_KEY_W);
            controls.steer    = keyAxis(GLFW_KEY_A, GLFW_KEY_D);
        }
        game.step(controls, dt);
    }
};

int main(int argc, char** argv)
{
    initSaigaSample();
    RacingSample sample(RacingSample::hasFlag(argc, argv, "--demo"));
    sample.runGame();
    return 0;
}
