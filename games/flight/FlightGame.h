/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/world/WorldRenderer.h"

#include "games/common/ChaseCamera.h"
#include "games/common/MeshRenderer.h"
#include "games/flight/FlightAutopilot.h"
#include "games/flight/FlightModel.h"

namespace Saiga
{
namespace Game
{
/**
 * The flight simulator tech demo (docs/engine_design.md §18.4): a small airplane over the 8 km world prototype
 * (WorldRenderer: CDLOD terrain, 200k GPU culled trees, grass) with a chase camera. The demo mode flies a circuit
 * with the autopilot. The sample (samples/vulkan/sample_vulkan_flight.cpp) maps the keyboard to the controls.
 */
class FlightGame
{
   public:
    void init(Vulkan::VulkanForwardRenderer& renderer, bool demo, int treeCount = 200000);
    void destroy();

    // Fixed step: the player's controls are used unless the demo mode is on
    void step(const AircraftControls& player, double dt);
    // Per frame: camera from the interpolated state (alpha between the last two steps)
    void transfer(vk::CommandBuffer cmd, float aspect, double frameDt, double alpha);
    void render(vk::CommandBuffer cmd);
    void imgui();

    void respawn();

    bool demo = false;
    const FlightModel& model() const { return plane; }
    Vulkan::WorldRenderer& world() { return worldRenderer; }

   private:
    Vulkan::WorldRenderer worldRenderer;
    MeshRenderer meshes;
    ChaseCamera camera;
    FlightModel plane;
    FlightAutopilot autopilot;
    AircraftState previous;
    AircraftControls lastControls;
    int planeMesh       = -1;
    double crashTimer   = 0;
    Vec3 renderPosition = Vec3(0, 0, 0);
    Quat renderRotation = Quat::Identity();
};

}  // namespace Game
}  // namespace Saiga
