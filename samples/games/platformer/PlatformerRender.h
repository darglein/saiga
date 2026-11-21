/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/app/EngineApp.h"

#include "games/common/ChaseCamera.h"
#include "games/common/GameEngine.h"
#include "games/common/MeshRenderer.h"
#include "games/platformer/CharacterAnimation.h"
#include "games/platformer/PlatformerWorld.h"

#include <array>

namespace Saiga
{
namespace Game
{
/**
 * The render side of the platformer (GameRenderHooks; transitional until S19): the level and the flag, every humanoid
 * built from boxes with the procedural animation (render-side until S34; one animator per character, landings come
 * from the step listener of the render caches; S16b: remote characters on a co-op client from RenderPose, their motion
 * read as late), the third-person camera (Q/E turn it; the demo's turns behind the character), the GUI.
 */
class PlatformerRender
{
   public:
    explicit PlatformerRender(std::shared_ptr<const PlatformerContext> context) : context(std::move(context)) {}

    void init(Engine::EngineApp& app);
    void shutdown();
    void transfer(vk::CommandBuffer cmd);
    void render(vk::CommandBuffer cmd);
    void gui();

   private:
    // The motion the animation reads (CharacterMotion of a step)
    struct Motion
    {
        Vec3 velocity       = Vec3::Zero();
        double yaw          = 0;
        bool grounded       = false;
        double airTime      = 0;
        bool jumped         = false;
        double landingSpeed = 0;
    };
    // A character as drawn this frame
    struct Character
    {
        Entity entity;
        CharacterAnimator animator;
        Vec3 position = Vec3::Zero();
        double yaw    = 0;
        bool drawn    = false;
    };
    Entity focus();
    Character& character(Entity e);

    std::shared_ptr<const PlatformerContext> context;
    Engine::EngineApp* app = nullptr;
    MeshRenderer meshes;
    ChaseCamera camera;
    RenderCaches<Motion> caches;
    std::vector<Character> characters;
    HeldCamera held;
    FrameClock clock;
    Entity followed;
    double cameraYaw = 0;
    double renderYaw = 0;  // the followed character's
    int levelMesh = -1, flagMesh = -1;
    int loadListener = -1;
    std::array<int, JointCount> partMesh;
};

}  // namespace Game
}  // namespace Saiga
