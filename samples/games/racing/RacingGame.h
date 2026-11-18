/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/world/WorldRenderer.h"

#include "games/common/ChaseCamera.h"
#include "games/common/MeshRenderer.h"
#include "games/racing/LapTimer.h"
#include "games/racing/RaceDriver.h"
#include "games/racing/Track.h"
#include "games/racing/Vehicle.h"

namespace Saiga
{
namespace Game
{
/**
 * The racing tech demo (docs/engine_design.md §18.4): a car with a simple vehicle model on a closed track over the
 * terrain of the world prototype (the track is placed on gentle land), checkpoints and lap times, a chase camera. The
 * demo mode lets the AI driver race.
 */
class RacingGame
{
   public:
    void init(Vulkan::VulkanForwardRenderer& renderer, bool demo, int treeCount = 200000);
    void destroy();

    void step(const VehicleControls& player, double dt);
    void transfer(vk::CommandBuffer cmd, float aspect, double frameDt, double alpha);
    void render(vk::CommandBuffer cmd);
    void imgui();
    // Back on the track at the last checkpoint
    void respawn();

    bool demo = false;

   private:
    MeshData trackMesh(const GroundHeight& ground) const;

    Vulkan::WorldRenderer worldRenderer;
    MeshRenderer meshes;
    ChaseCamera camera;
    Track track;
    Vehicle car;
    RaceDriver driver;
    LapTimer timer;
    VehicleState previous;
    VehicleControls lastControls;
    int bodyMesh = -1, wheelMesh = -1, trackMeshId = -1, coneMesh = -1, bannerMesh = -1;
    Vec3 renderPosition = Vec3(0, 0, 0);
    Quat renderRotation = Quat::Identity();
    // Wheel visuals interpolated like the body pose
    std::array<double, 4> renderCompression = {}, renderSpin = {};
    double renderSteer = 0;
    double upsideDown  = 0;
    GroundHeight terrain;
    // Arc length near the car, speeds up the track surface queries of the wheels
    double surfaceHint = -1;
};

}  // namespace Game
}  // namespace Saiga
