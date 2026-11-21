/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The render side of the flight demo: world prototype, airplane, chase camera, GUI.

#include "games/flight/FlightRender.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/engine/gameplay/PlayerAction.h"
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
    // The context's height field, shared (the aliasing pointer keeps the context alive); RenderScene draws the
    // prototype world and the trees (S19a)
    worldRenderer.init(params, std::shared_ptr<const Vulkan::HeightField>(context, &context->heightField));
    worldRenderer.attach(*app->scene());
    meshes.init(app->scene()->base(), app->scene()->formats());
    planeMesh = meshes.addMesh(MeshBuilder::airplane());

    const FlightContext* ctx = context.get();
    camera.distance          = 24;
    camera.height            = 5;
    camera.targetHeight      = 1;
    camera.stiffness         = 4;
    camera.ground            = [ctx](double x, double z) { return ctx->ground(x, z); };
    caches.attach(
        app->world(), [](World& w, const std::function<void(Entity)>& fn)
        { w.view<const AircraftState>().each([&](Entity e, const AircraftState&) { fn(e); }); },
        [](const World&, Entity) { return NoState(); });
}

void FlightRender::shutdown()
{
    caches.detach();
    meshes.destroy();
    worldRenderer.destroy();
}

Entity FlightRender::focus()
{
    World& w = app->world();
    // The local player is a driver: its plane
    const Entity mine = drivenVehicle(w, app->localPlayer());
    if (mine.valid() && w.has<AircraftState>(mine)) return mine;
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
    // The scene pass of this frame: linear and pre-exposed with a tone map, display values in clay mode (S17)
    Vulkan::RenderScene& scene = *app->scene();
    worldRenderer.update();
    meshes.setOutputLinear(scene.sceneWritesLinear(), scene.preExposure());
    const double frameDt = clock.tick();
    World& w             = app->world();
    const Entity e       = focus();
    bool snap            = false;
    if (e != followed)
    {
        followed = e;
        snap     = true;
    }
    // Every plane at its interpolated pose
    drawn.clear();
    PoseHistory::Pose focusPose;
    bool hasFocus = false;
    w.view<const AircraftState>().each(
        [&](Entity plane, const AircraftState&)
        {
            auto* c = caches.entry(plane);
            PoseHistory::Pose p;
            if (!c || !c->pose.interpolated(w, plane, w.alpha(), p)) return;
            const bool teleported = c->pose.takeTeleport();
            if (plane == followed)
            {
                focusPose = p;
                hasFocus  = true;
                snap      = snap || teleported;
            }
            drawn.push_back(p);
        });
    const bool toolCamera = app->cameraHeldByTool();
    const bool released   = held.released(toolCamera);
    if (snap || released) camera.reset();
    if (hasFocus && !toolCamera) camera.update(focusPose.position, forwardOf(focusPose.rotation), frameDt);

    // A tool holds the camera (S14): the app camera, else the chase camera
    const mat4 view = toolCamera ? app->camera().view : camera.view();
    const Vec3 eye  = toolCamera ? Vec3(app->camera().getPosition().cast<double>()) : camera.eye();
    if (!toolCamera) app->camera().setView(view);
    // RenderScene's camera of this frame and the time of the wind (the world's time between the steps)
    scene.setCamera(0, eye, Vulkan::viewRotation(view), radians(double(worldRenderer.settings.fovy)));
    scene.setTime(w.time() + w.alpha() * w.dt());
    meshes.transfer(cmd, scene.viewMatrix(), eye, scene.projectionMatrix(), worldLighting(worldRenderer));
}

void FlightRender::render(vk::CommandBuffer cmd)
{
    for (const PoseHistory::Pose& p : drawn) meshes.draw(planeMesh, transformMatrix(p.position, p.rotation));
    meshes.render(cmd);
}

void FlightRender::gui()
{
    worldRenderer.imgui();
    // The GUI runs before transfer() in a frame: the plane it shows is focus()'s
    ImGui::SetNextWindowPos(ImVec2(10, ImGui::GetIO().DisplaySize.y - 160), ImGuiCond_FirstUseEver);
    ImGui::Begin("Flight", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Text("W/S pitch, A/D roll, Q/E yaw, shift/ctrl throttle, R respawn, G GUI");
    World& w       = app->world();
    const Entity e = focus();
    if (w.alive(e) && w.has<AircraftState>(e))
    {
        const FlightModel m    = flightModelOf(w, e);
        const AircraftState& s = w.read<AircraftState>(e);
        const bool mine        = drivenVehicle(w, app->localPlayer()) == e;
        const bool autopilot   = w.has<VehicleSeat>(e) && w.read<VehicleSeat>(e).driver == e;
        if (app->coop())
            ImGui::Text("co-op client (peer %d): %s, %d planes", int(w.localPeer()),
                        mine ? "your plane" : (autopilot ? "the host's autopilot" : "another player's plane"),
                        int(w.registry().poolById(AircraftState::id)->size()));
        else
            ImGui::Text("%s", mine ? "player" : "demo (autopilot)");
        ImGui::Text("airspeed %.1f m/s, altitude %.0f m above ground", m.airspeed(), m.altitudeAboveGround());
        ImGui::Text("throttle %.2f, angle of attack %.1f deg, bank %.0f deg", s.throttle, m.angleOfAttack() * 180 / kPi,
                    m.bankAngle() * 180 / kPi);
        if (m.stalled()) ImGui::TextColored(ImVec4(1, 0.5f, 0.2f, 1), "STALL");
        if (s.crashed) ImGui::TextColored(ImVec4(1, 0.2f, 0.2f, 1), "CRASHED - respawn in a moment");
        // The local player's action (a message on a co-op client; it acts when steps run); without a local player
        // (--demo on the host) the game object's respawn directly
        const Entity player = app->localPlayer();
        if (mine)
        {
            if (ImGui::Button(w.isPaused() ? "respawn (when steps run)" : "respawn"))
                w.queueAction(PlayerAction{player, uint32_t(FlightAction::Respawn), Vec3::Zero()});
        }
        else if (w.isHost() && !player.valid() && ImGui::Button("respawn"))
        {
            if (auto* plane = gameObjectAs<AircraftBase>(w, e)) plane->respawn();
        }
    }
    ImGui::End();
}

}  // namespace Game
}  // namespace Saiga
