/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The flight simulator tech demo: world prototype, airplane, chase camera.

#include "games/flight/FlightGame.h"

#include "saiga/core/imgui/imgui.h"

#include "games/common/WorldLighting.h"

namespace Saiga
{
namespace Game
{
void FlightGame::init(Vulkan::VulkanForwardRenderer& renderer, bool _demo, int treeCount)
{
    demo = _demo;
    Vulkan::WorldParameters params;
    params.treeCount = treeCount;
    worldRenderer.init(renderer, params);
    meshes.init(renderer);
    planeMesh = meshes.addMesh(MeshBuilder::airplane());

    plane.ground        = worldGround(worldRenderer, true);
    double size         = worldRenderer.heightField().worldSize();
    autopilot.center    = Vec3(size / 2, 0, size / 2);
    autopilot.radius    = size * 0.3;
    camera.distance     = 24;
    camera.height       = 5;
    camera.targetHeight = 1;
    camera.stiffness    = 4;
    camera.ground       = plane.ground;
    respawn();

    if (demo)
    {
        // Fly a while before the first frame, so the demo starts in a turn (and SAIGA_SCREENSHOT shows it)
        for (int i = 0; i < 60 * 12; ++i) step(AircraftControls(), 1.0 / 60);
    }
}

void FlightGame::destroy()
{
    meshes.destroy();
    worldRenderer.destroy();
}

void FlightGame::respawn()
{
    // On the autopilot's circuit, heading along it, 300 m above the ground
    Vec3 p = autopilot.center + Vec3(autopilot.radius, 0, 0);
    p(1)   = plane.ground(p(0), p(2)) + 300;
    plane.reset(p, headingOf(Vec3(0, 0, 1)), 65);
    previous   = plane.state;
    crashTimer = 0;
    // The chase camera jumps along in the next frame (it would glide from the crash site otherwise)
    camera.reset();
}

void FlightGame::step(const AircraftControls& player, double dt)
{
    previous = plane.state;
    if (plane.state.crashed)
    {
        crashTimer += dt;
        if (crashTimer > 2.0) respawn();
        return;
    }
    lastControls = demo ? autopilot.controls(plane) : player;
    plane.step(lastControls, dt);
}

void FlightGame::transfer(vk::CommandBuffer cmd, float aspect, double frameDt, double alpha)
{
    renderPosition = previous.position + (plane.state.position - previous.position) * alpha;
    renderRotation = previous.orientation.slerp(alpha, plane.state.orientation);
    camera.update(renderPosition, forwardOf(renderRotation), frameDt);

    mat4 view = camera.view();
    worldRenderer.transfer(cmd, view, camera.eye().cast<float>(), aspect);
    meshes.transfer(cmd, view, camera.eye(), worldRenderer.projection(aspect), worldLighting(worldRenderer));
}

void FlightGame::render(vk::CommandBuffer cmd)
{
    worldRenderer.render(cmd);
    meshes.draw(planeMesh, transformMatrix(renderPosition, renderRotation));
    meshes.render(cmd);
}

void FlightGame::imgui()
{
    worldRenderer.imgui();
    ImGui::SetNextWindowPos(ImVec2(10, ImGui::GetIO().DisplaySize.y - 160), ImGuiCond_FirstUseEver);
    ImGui::Begin("Flight");
    ImGui::Text("W/S pitch, A/D roll, Q/E yaw, shift/ctrl throttle, R respawn, G GUI");
    ImGui::Checkbox("demo (autopilot)", &demo);
    ImGui::Text("airspeed %.1f m/s, altitude %.0f m above ground", plane.airspeed(), plane.altitudeAboveGround());
    ImGui::Text("throttle %.2f, angle of attack %.1f deg, bank %.0f deg", lastControls.throttle,
                plane.angleOfAttack() * 180 / kPi, plane.bankAngle() * 180 / kPi);
    if (plane.stalled()) ImGui::TextColored(ImVec4(1, 0.5f, 0.2f, 1), "STALL");
    if (plane.state.crashed) ImGui::TextColored(ImVec4(1, 0.2f, 0.2f, 1), "CRASHED - respawn in a moment");
    if (ImGui::Button("respawn")) respawn();
    ImGui::End();
}

}  // namespace Game
}  // namespace Saiga
