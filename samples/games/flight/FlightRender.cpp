/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The render side of the flight demo: the prototype environment, chase camera, GUI (the airplanes, the terrain and the
// generated content are drawn by the engine).

#include "games/flight/FlightRender.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/engine/gameplay/PlayerAction.h"
#include "saiga/engine/terrain/TerrainSystem.h"
#include "saiga/vulkan/scene/RenderScene.h"

#include "games/common/Mesh.h"
#include "games/common/RenderAssets.h"

#include <algorithm>
#include <thread>


namespace Saiga
{
namespace Game
{
using namespace Engine;

void FlightRender::init(EngineApp& _app)
{
    app = &_app;
    // The terrain, its grass and the trees, rocks and ruins are the engine's (the World's generated space through the
    // render sink: S20's terrain, S23b's content sets of the OverworldGenerator); PrototypeScene adds the environment
    // (sky, water, the foliage texture of the tree meshes' seed) and no forest of its own (S23c)
    prototype.attach(*app->scene(), {}, kFlightFoliageSeed);
    // The world prototype's settings and tree counts as a section of the debug UI (S19c)
    app->debugUi().addSection(DebugTab::Engine, "World", [this]() { prototype.imgui(); });

    // The content sets of the 3 km radius (flightSpaceTemplate) are generated on the terrain's streaming pool: more
    // workers than the default 2 (a third of the hardware threads, 2 to 6) keep the first frame (completeView) and the
    // streaming at cruise speed fast; set before the pool is created by the first frame
    TerrainViewSettings streaming = app->world().terrain().viewSettings();
    streaming.workers             = std::clamp(int(std::thread::hardware_concurrency()) / 3, 2, 6);
    app->world().terrain().setViewSettings(streaming);

    camera.distance     = 24;
    camera.height       = 5;
    camera.targetHeight = 1;
    camera.stiffness    = 4;
    camera.ground       = [this](double x, double z) { return flightGround(app->world(), x, z); };
}

void FlightRender::shutdown()
{
    removeLights();
    prototype.detach();
}

void FlightRender::removeLights()
{
    for (auto& p : planeLights)
    {
        for (Vulkan::LightHandle h : p.second.lights)
            if (h) app->scene()->removeLight(h);
        for (Vulkan::InstanceHandle h : p.second.bulbs)
            if (h) app->scene()->removeDynamic(h);
    }
    planeLights.clear();
}

void FlightRender::updateLights(double alpha)
{
    Vulkan::RenderScene& scene = *app->scene();
    World& w                   = app->world();
    // Game policy: the lights are on while the sun is below 5 degrees
    if (!(app->sampleViewEnvironment().celestial.sunElevationDeg < 5.0))
    {
        removeLights();
        return;
    }
    // The light of index k of a plane: offset (body frame: forward -z, right +x), colour, lumen, range, spot axis
    struct Spec
    {
        Vec3 offset;
        Vec3 color;
        float lumen, range;
        bool spot;
    };
    static const std::array<Spec, 4> specs = {Spec{Vec3(-4.85, 0.1, -0.7), Vec3(1.0, 0.08, 0.05), 120, 7, false},
                                              Spec{Vec3(4.85, 0.1, -0.7), Vec3(0.1, 1.0, 0.25), 120, 7, false},
                                              Spec{Vec3(0, 1.75, 4.7), Vec3(1, 1, 1), 80, 6, false},
                                              Spec{Vec3(0, -0.35, -3.6), Vec3(1.0, 0.95, 0.85), 20000, 120, true}};
    // The bulbs' meshes once (emissive: they glow at night)
    if (!bulbMesh)
    {
        const Vulkan::MaterialId m0 = Vulkan::MaterialId(scene.reserveIds(Vulkan::SceneIdKind::Material, 3));
        bulbMesh                    = Vulkan::MeshId(scene.reserveIds(Vulkan::SceneIdKind::Mesh, 3));
        for (int k = 0; k < 3; ++k)
        {
            Vulkan::MaterialData m = demoMaterial();
            m.emissive             = specs[size_t(k)].color.cast<float>();
            const std::string name = "flight/nav_bulb" + std::to_string(k);
            scene.addMaterial(Vulkan::MaterialId(m0 + k), name, m);
            scene.addMesh(Vulkan::MeshId(bulbMesh + k), name,
                          *meshLods(MeshBuilder::box(Vec3(0.09, 0.09, 0.09), specs[size_t(k)].color), 3000),
                          Vulkan::MaterialId(m0 + k));
        }
    }
    std::map<uint64_t, PlaneLights> seen;
    w.view<const AircraftState>().each(
        [&](Entity e, const AircraftState&)
        {
            Engine::WorldPose p;
            if (!w.extract().interpolatedPose(e, alpha, p)) return;
            const uint64_t key = (uint64_t(e.index) << 32) | e.generation;
            auto it            = planeLights.find(key);
            PlaneLights pl;
            if (it != planeLights.end())
            {
                pl = it->second;
                planeLights.erase(it);
            }
            std::array<Vulkan::LightHandle, 4>& handles = pl.lights;
            for (size_t k = 0; k < specs.size(); ++k)
            {
                const Spec& s = specs[k];
                Engine::WorldPose at;
                at.space          = p.space;
                at.position       = p.position + p.rotation * s.offset;
                const ChunkPose c = toChunkPose(at);
                Vulkan::LightData d;
                d.kind     = s.spot ? Vulkan::LightKind::Spot : Vulkan::LightKind::Point;
                d.position = c.pose.position.cast<float>();
                // The landing light: forward and 18 degrees down (the ground ahead below about 100 m)
                d.direction   = (p.rotation * Vec3(0, -0.325, -1).normalized()).cast<float>();
                d.color       = s.color.cast<float>();
                d.lumen       = s.lumen;
                d.range       = s.range;
                d.cosInner    = float(std::cos(14.0 * kPi / 180));
                d.cosOuter    = float(std::cos(24.0 * kPi / 180));
                d.castShadows = s.spot;
                d.priority    = s.spot ? 4.0f : 1.0f;
                if (handles[k])
                    scene.setLight(handles[k], c.chunk, d);
                else
                    handles[k] = scene.addLight(c.chunk, d);
                if (k < 3)
                {
                    // The bulb at the light (snapped: the pose is already interpolated)
                    Vulkan::InstanceHandle& b = pl.bulbs[k];
                    if (!b) b = scene.addDynamic(Vulkan::MeshId(bulbMesh + k));
                    Vulkan::DynamicPose pose = c.pose;
                    pose.rotation            = p.rotation;
                    pose.scale               = 1;
                    scene.setDynamic(b, c.chunk, pose, pose);
                }
            }
            seen[key] = pl;
        });
    // The planes that are gone
    removeLights();
    planeLights = std::move(seen);
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
        cameraSpace = p.space;
    }
    updateLights(alpha);
    // While a tool holds the camera the app ignores it (S14)
    app->setViewCamera(cameraSpace, camera.eye(), Vulkan::viewRotation(camera.view()),
                       radians(double(prototype.settings.fovy)));
}

void FlightRender::gui()
{
    // The GUI runs inside the frame (after beforeRender): the plane it shows is focus()'s
    ImGui::SetNextWindowPos(ImVec2(10, ImGui::GetIO().DisplaySize.y - 160), ImGuiCond_FirstUseEver);
    ImGui::Begin("Flight", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Text("W/S pitch, A/D roll, Q/E yaw, shift/ctrl throttle, R respawn, F1-F5 debug tabs, G GUI");
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
