/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The render side of the flight demo: world prototype, airplane, chase camera, GUI.

#include "games/flight/FlightRender.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/vulkan/scene/RenderScene.h"

#include "games/common/WorldLighting.h"

namespace Saiga
{
namespace Game
{
using namespace Engine;

void FlightRender::init(EngineApp& _app)
{
    app = &_app;
    Vulkan::WorldParameters params;
    params.treeCount = worldTreeCount();
    // The context's height field, shared (the aliasing pointer keeps the context alive)
    worldRenderer.init(app->scene()->forwardRenderer(), params,
                       std::shared_ptr<const Vulkan::HeightField>(context, &context->heightField));
    meshes.init(app->scene()->forwardRenderer());
    planeMesh = meshes.addMesh(MeshBuilder::airplane());

    const FlightContext* ctx = context.get();
    camera.distance          = 24;
    camera.height            = 5;
    camera.targetHeight      = 1;
    camera.stiffness         = 4;
    camera.ground            = [ctx](double x, double z) { return ctx->ground(x, z); };
    pose.attach(app->world());
}

void FlightRender::shutdown()
{
    pose.detach();
    meshes.destroy();
    worldRenderer.destroy();
}

Entity FlightRender::focus()
{
    World& w = app->world();
    Entity p = app->localPlayer();
    if (p.valid() && w.alive(p) && w.has<AircraftState>(p)) return p;
    Entity first;
    w.view<const AircraftState>().each(
        [&](Entity e, const AircraftState&)
        {
            if (!first.valid()) first = e;
        });
    return first;
}

void FlightRender::transfer(vk::CommandBuffer cmd)
{
    const double frameDt = clock.tick();
    const ivec2 size     = app->scene()->size();
    const float aspect   = float(size(0)) / float(std::max(1, size(1)));
    pose.follow(focus());
    hasPose = pose.interpolated(app->world().alpha(), renderPose);
    if (pose.takeTeleport()) camera.reset();
    if (hasPose) camera.update(renderPose.position, forwardOf(renderPose.rotation), frameDt);

    const mat4 view = camera.view();
    app->camera().setView(view);
    worldRenderer.transfer(cmd, view, camera.eye().cast<float>(), aspect);
    meshes.transfer(cmd, view, camera.eye(), worldRenderer.projection(aspect), worldLighting(worldRenderer));
}

void FlightRender::render(vk::CommandBuffer cmd)
{
    worldRenderer.render(cmd);
    if (hasPose) meshes.draw(planeMesh, transformMatrix(renderPose.position, renderPose.rotation));
    meshes.render(cmd);
}

void FlightRender::gui()
{
    worldRenderer.imgui();
    // The GUI runs before transfer() in a frame: follow here too (the first frame's window had no fields)
    pose.follow(focus());
    ImGui::SetNextWindowPos(ImVec2(10, ImGui::GetIO().DisplaySize.y - 160), ImGuiCond_FirstUseEver);
    ImGui::Begin("Flight", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Text("W/S pitch, A/D roll, Q/E yaw, shift/ctrl throttle, R respawn, G GUI");
    World& w       = app->world();
    const Entity e = pose.followed();
    if (w.alive(e) && w.has<AircraftState>(e))
    {
        const FlightModel m       = flightModelOf(w, e);
        const AircraftState& s    = w.read<AircraftState>(e);
        const AircraftControls& c = w.read<AircraftControls>(e);
        ImGui::Text("%s", app->localPlayer() == e ? "player" : "demo (autopilot)");
        ImGui::Text("airspeed %.1f m/s, altitude %.0f m above ground", m.airspeed(), m.altitudeAboveGround());
        ImGui::Text("throttle %.2f, angle of attack %.1f deg, bank %.0f deg", c.throttle, m.angleOfAttack() * 180 / kPi,
                    m.bankAngle() * 180 / kPi);
        if (m.stalled()) ImGui::TextColored(ImVec4(1, 0.5f, 0.2f, 1), "STALL");
        if (s.crashed) ImGui::TextColored(ImVec4(1, 0.2f, 0.2f, 1), "CRASHED - respawn in a moment");
        if (ImGui::Button("respawn") && app->localPlayer() == e)
            w.queueAction(PlayerAction{e, uint32_t(FlightAction::Respawn), Vec3::Zero()});
    }
    ImGui::End();
}

}  // namespace Game
}  // namespace Saiga
