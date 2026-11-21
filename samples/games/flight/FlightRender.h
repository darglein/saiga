/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/app/EngineApp.h"
#include "saiga/vulkan/world/WorldRenderer.h"

#include "games/common/ChaseCamera.h"
#include "games/common/GameEngine.h"
#include "games/common/MeshRenderer.h"
#include "games/flight/FlightWorld.h"

namespace Saiga
{
namespace Game
{
/**
 * The render side of the flight demo (GameRenderHooks; transitional until the RenderScene instances of S19): the
 * world prototype on the context's height field, every airplane at its pose interpolated between the last two steps
 * (S16b: remote planes on a co-op client from RenderPose), the chase camera, the GUI. Follows the local player's plane
 * (its driver's Driver.vehicle), or the first airplane (--demo).
 */
class FlightRender
{
   public:
    explicit FlightRender(std::shared_ptr<const FlightContext> context) : context(std::move(context)) {}

    void init(Engine::EngineApp& app);
    void shutdown();
    void transfer(vk::CommandBuffer cmd);
    void render(vk::CommandBuffer cmd);
    void gui();

   private:
    struct NoState
    {
    };
    Entity focus();

    std::shared_ptr<const FlightContext> context;
    Engine::EngineApp* app = nullptr;
    Vulkan::WorldRenderer worldRenderer;
    MeshRenderer meshes;
    ChaseCamera camera;
    RenderCaches<NoState> caches;
    HeldCamera held;
    FrameClock clock;
    Entity followed;
    std::vector<PoseHistory::Pose> drawn;
    int planeMesh = -1;
};

}  // namespace Game
}  // namespace Saiga
