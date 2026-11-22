/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The render side of the flight demo: world prototype, chase camera, GUI (the airplanes are drawn by the engine).

#include "games/flight/FlightRender.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/engine/gameplay/PlayerAction.h"
#include "saiga/vulkan/scene/RenderScene.h"


namespace Saiga
{
namespace Game
{
using namespace Engine;

void FlightRender::init(EngineApp& _app)
{
    app = &_app;
    // The terrain is the engine's (the World's space through the render sink, S20); the trees of the prototype's 8 km
    // area and the environment on the scene
    Vulkan::PrototypeArea area;
    area.seed        = context->seed;
    area.baseSpacing = context->baseSpacing;
    prototype.attach(
        *app->scene(),
        Vulkan::ForestRenderer::placeTrees(area, worldTreeCount(), 1, Vulkan::FoliageGenerator::treeSpecies(1)), 1);
    // The world prototype's settings and tree counts as a section of the debug UI (S19c)
    app->debugUi().addSection(DebugTab::Engine, "World", [this]() { prototype.imgui(); });

    const FlightContext* ctx = context.get();
    camera.distance          = 24;
    camera.height            = 5;
    camera.targetHeight      = 1;
    camera.stiffness         = 4;
    camera.ground            = [ctx](double x, double z) { return ctx->ground(x, z); };
}

void FlightRender::shutdown()
{
    prototype.detach();
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

void FlightRender::beforeRender(double alpha)
{
    Vulkan::RenderScene& scene = *app->scene();
    prototype.update();
    const double frameDt = clock.tick();
    World& w             = app->world();
    const Entity e       = focus();
    if (e != followed)
    {
        followed = e;
        camera.reset();
    }
    // The chase camera behind the interpolated plane (the engine draws it with the same pose); a teleport (respawn)
    // snaps it once (the extract's snap sequence changes once per snap)
    Engine::WorldPose p;
    if (w.extract().interpolatedPose(followed, alpha, p))
    {
        const uint64_t snaps = w.extract().snapSequence(followed);
        if (snaps != followedSnaps)
        {
            followedSnaps = snaps;
            camera.reset();
        }
        camera.update(p.position, forwardOf(p.rotation), frameDt);
    }
    // While a tool holds the camera the app ignores it (S14)
    app->setViewCamera(0, camera.eye(), Vulkan::viewRotation(camera.view()), radians(double(prototype.settings.fovy)));
    // The time of the wind (the world's time between the steps)
    scene.setTime(w.time() + alpha * w.dt());
}

void FlightRender::gui()
{
    // The GUI runs inside the frame (after beforeRender): the plane it shows is focus()'s
    ImGui::SetNextWindowPos(ImVec2(10, ImGui::GetIO().DisplaySize.y - 160), ImGuiCond_FirstUseEver);
    ImGui::Begin("Flight", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Text("W/S pitch, A/D roll, Q/E yaw, shift/ctrl throttle, R respawn, F1-F4 debug tabs, G GUI");
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
            // Between steps, not while the frame is recorded (S19c)
            app->queueBetweenSteps(
                [this, e]()
                {
                    World& world = app->world();
                    if (!world.alive(e)) return;
                    if (auto* plane = gameObjectAs<AircraftBase>(world, e)) plane->respawn();
                });
        }
    }
    ImGui::End();
}

}  // namespace Game
}  // namespace Saiga
