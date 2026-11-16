/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

/**
 * Platformer tech demo (games/platformer, docs/engine_design.md §18.4): one level, a procedurally animated character.
 * --demo: the character runs through the level by itself.
 *
 * Controls: WASD move (relative to the camera), shift walk, space jump, Q/E turn the camera, R restart, G GUI, ESC.
 */

#include "games/common/GameInput.h"
#include "games/common/GameSampleHost.h"
#include "games/platformer/PlatformerGame.h"

using namespace Saiga;
using namespace Saiga::Game;

class PlatformerSample : public GameSampleHost<PlatformerGame>
{
   public:
    explicit PlatformerSample(bool demo) : GameSampleHost(vec4(0.62f, 0.74f, 0.86f, 1)) { game.init(*renderer, demo); }

   private:
    void stepGame(double dt, bool keyboardFree) override
    {
        PlatformerGame::PlayerInput in;
        // The edge history follows the key also while the GUI has the keyboard: otherwise a release during the capture
        // is missed and the next press after it is not seen as a new press
        bool jump        = keyDown(GLFW_KEY_SPACE);
        bool jumpPressed = jump && !jumpWasDown;
        jumpWasDown      = jump;
        if (keyboardFree)
        {
            in.forward     = keyAxis(GLFW_KEY_S, GLFW_KEY_W);
            in.right       = keyAxis(GLFW_KEY_A, GLFW_KEY_D);
            in.walk        = keyDown(GLFW_KEY_LEFT_SHIFT);
            in.jumpHeld    = jump;
            in.jumpPressed = jumpPressed;
            in.cameraTurn  = keyAxis(GLFW_KEY_E, GLFW_KEY_Q) * 2.0;
        }
        game.step(in, dt);
    }

    bool jumpWasDown = false;
};

int main(int argc, char** argv)
{
    initSaigaSample();
    PlatformerSample sample(PlatformerSample::hasFlag(argc, argv, "--demo"));
    sample.runGame();
    return 0;
}
