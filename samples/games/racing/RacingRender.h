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
#include "games/racing/RacingWorld.h"

namespace Saiga
{
namespace Game
{
/**
 * The render side of the racing demo (GameRenderHooks; transitional until S19): the world prototype on the context's
 * height field (no grass: it would grow through the track), the track, gates and banner, the car with its wheels at
 * the pose interpolated between the last two steps (the wheels' compression, spin and steering too), the chase camera,
 * the GUI. Follows the local player, or the first car (--demo).
 */
class RacingRender
{
   public:
    explicit RacingRender(std::shared_ptr<const RacingContext> context) : context(std::move(context)) {}

    void init(Engine::EngineApp& app);
    void shutdown();
    void transfer(vk::CommandBuffer cmd);
    void render(vk::CommandBuffer cmd);
    void gui();

   private:
    struct Wheels
    {
        std::array<double, 4> compression = {}, spin = {};
        double steer = 0;
    };
    Entity focus();
    MeshData trackMesh() const;

    std::shared_ptr<const RacingContext> context;
    Engine::EngineApp* app = nullptr;
    Vulkan::WorldRenderer worldRenderer;
    MeshRenderer meshes;
    ChaseCamera camera;
    PoseHistory pose;
    HeldCamera held;
    FrameClock clock;
    PoseHistory::Pose renderPose;
    bool hasPose = false;
    // The wheels of the followed car after the last two steps (a step listener) and interpolated
    Wheels previousWheels, currentWheels, renderWheels;
    int bodyMesh = -1, wheelMesh = -1, trackMeshId = -1, coneMesh = -1, bannerMesh = -1;
    int stepListener = -1, loadListener = -1;
    // Both wheel sets from the followed car's VehicleState (a new car, a load)
    void reseedWheels();
};

}  // namespace Game
}  // namespace Saiga
