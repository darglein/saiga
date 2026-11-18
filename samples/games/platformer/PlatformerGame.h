/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "games/common/ChaseCamera.h"
#include "games/common/MeshRenderer.h"
#include "games/platformer/CharacterAnimation.h"
#include "games/platformer/CharacterController.h"
#include "games/platformer/PlatformerDemo.h"

namespace Saiga
{
namespace Game
{
/**
 * The platformer tech demo (docs/engine_design.md §18.4): one level of platforms, a humanoid built from boxes with
 * procedural skeletal animation (CharacterAnimator), a kinematic character controller and a third-person camera. The
 * demo mode runs through the level (PlatformerDemo).
 */
class PlatformerGame
{
   public:
    struct PlayerInput
    {
        // Relative to the camera: x right, y forward
        double right = 0, forward = 0;
        bool walk = false, jumpHeld = false, jumpPressed = false;
        // Camera orbit (rad/s)
        double cameraTurn = 0;
    };

    void init(Vulkan::VulkanForwardRenderer& renderer, bool demo);
    void destroy();

    void step(const PlayerInput& input, double dt);
    void transfer(vk::CommandBuffer cmd, float aspect, double frameDt, double alpha);
    void render(vk::CommandBuffer cmd);
    void imgui();
    void respawn();

    bool demo = false;

   private:
    // Advances the animation by dt from a (possibly interpolated) state
    void animate(const CharacterState& state, double dt);

    Vulkan::VulkanForwardRenderer* renderer = nullptr;
    MeshRenderer meshes;
    ChaseCamera camera;
    Level level;
    CharacterController character;
    CharacterAnimator animator;
    PlatformerDemo demoRunner;
    CharacterState previous;
    double pendingLanding = 0;
    double cameraYaw      = 0;
    double goalTime = -1, runTime = 0;

    int levelMesh = -1, flagMesh = -1;
    std::array<int, JointCount> partMesh;
    std::array<Mat4, JointCount> partOffset;
    Vec3 renderFeet   = Vec3(0, 0, 0);
    double renderYaw  = 0;
    float aspectRatio = 1;
};

}  // namespace Game
}  // namespace Saiga
