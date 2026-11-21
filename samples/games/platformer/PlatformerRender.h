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
 * The render side of the platformer (GameRenderHooks; transitional until S19): the level and the flag, the humanoid
 * built from boxes with the procedural animation (render-side until S34; landings come from a step listener), the
 * third-person camera (Q/E turn it; the demo's turns behind the character), the GUI.
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
    Entity focus();

    std::shared_ptr<const PlatformerContext> context;
    Engine::EngineApp* app = nullptr;
    MeshRenderer meshes;
    ChaseCamera camera;
    CharacterAnimator animator;
    PoseHistory pose;
    HeldCamera held;
    FrameClock clock;
    PoseHistory::Pose renderPose;
    bool hasPose          = false;
    double cameraYaw      = 0;
    double pendingLanding = 0;
    double renderYaw      = 0;
    int levelMesh = -1, flagMesh = -1;
    int stepListener = -1, loadListener = -1;
    std::array<int, JointCount> partMesh;
};

}  // namespace Game
}  // namespace Saiga
