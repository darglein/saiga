/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The render side of the racing demo: world prototype, track, wheels, chase camera, GUI (the car bodies are drawn by
// the engine).

#include "games/racing/RacingRender.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/engine/components/Player.h"
#include "saiga/engine/gameplay/PlayerAction.h"
#include "saiga/engine/sync/VulkanRenderSink.h"
#include "saiga/vulkan/scene/RenderScene.h"


namespace Saiga
{
namespace Game
{
using namespace Engine;

void RacingRender::init(EngineApp& _app)
{
    app = &_app;
    Vulkan::WorldParameters params;
    // The renderer's trees may follow SAIGA_WORLD_TREES; the track's site does not (RacingContext::kSiteTreeCount)
    params.treeCount = worldTreeCount();
    worldRenderer.init(params, std::shared_ptr<const Vulkan::HeightField>(context, &context->heightField));
    // Grass would grow through the track
    worldRenderer.settings.grass = false;
    // RenderScene draws the prototype world and the trees (S19a)
    worldRenderer.attach(*app->scene());
    // The track, its cones and the banner: one instance set at the track's start (S19b)
    level.create(*app->scene(), "racing/level", levelMesh(), context->track.position(0));
    wheels.init(app->vulkanSink());

    const RacingContext* ctx = context.get();
    camera.distance          = 9;
    camera.height            = 3;
    camera.targetHeight      = 1.2;
    camera.stiffness         = 6;
    camera.ground            = [ctx](double x, double z) { return ctx->terrain(x, z); };
    // Every car's wheels (a load starts them over: no interpolation from the state before)
    caches.attach(
        app->world(), [](World& w, const std::function<void(Entity)>& fn)
        { w.view<const VehicleState>().each([&](Entity e, const VehicleState&) { fn(e); }); },
        [](const World& w, Entity e)
        {
            const VehicleState& s = w.read<VehicleState>(e);
            return Wheels{s.compression, s.spin, s.steerAngle};
        });
}

void RacingRender::shutdown()
{
    caches.detach();
    wheels.clear();
    level.release();
    worldRenderer.destroy();
}

MeshData RacingRender::levelMesh() const
{
    const Track& track = context->track;
    MeshData m;
    double half = track.width * 0.5, curb = track.curbWidth;
    Vec3 asphalt(0.32, 0.32, 0.34), red(0.8, 0.15, 0.12), white(0.92, 0.92, 0.92);
    int n = int(track.length() / 2.0);
    // A point of the surface: lifted asphalt, the curbs' outer edges on the terrain (as Track::surfaceOffset)
    auto edge = [&](double s, double offset, double lift)
    {
        Vec3 c = track.position(s), t = track.tangent(s);
        Vec3 r(-t(2), 0, t(0));
        Vec3 p = c + r * offset;
        p(1)   = context->terrain(p(0), p(2)) + lift;
        return p;
    };
    // The asphalt in 3 m wide strips, so it follows the terrain across the track like the car's ground does
    const int strips = std::max(1, int(std::ceil(track.width / 3.0)));
    for (int i = 0; i < n; ++i)
    {
        double s0 = track.length() * i / n, s1 = track.length() * (i + 1) / n;
        for (int k = 0; k < strips; ++k)
        {
            double a = -half + track.width * k / strips, b = -half + track.width * (k + 1) / strips;
            m.addQuad(edge(s0, a, track.lift), edge(s0, b, track.lift), edge(s1, b, track.lift),
                      edge(s1, a, track.lift), asphalt);
        }
        Vec3 cc = (i / 2) % 2 ? red : white;
        m.addQuad(edge(s0, -half - curb, 0), edge(s0, -half, track.lift), edge(s1, -half, track.lift),
                  edge(s1, -half - curb, 0), cc);
        m.addQuad(edge(s0, half, track.lift), edge(s0, half + curb, 0), edge(s1, half + curb, 0),
                  edge(s1, half, track.lift), cc);
    }
    // Cones at the gates, the banner at the start: two posts and a checkered bar
    for (auto& g : track.checkpoints())
    {
        Vec3 r(-g.tangent(2), 0, g.tangent(0));
        for (double side : {-1.0, 1.0})
        {
            Vec3 p = g.center + r * (side * (track.width * 0.5 + 1.5));
            p(1)   = context->terrain(p(0), p(2));
            m.append(MeshBuilder::trafficCone(), transformMatrix(p, Quat::Identity()));
        }
    }
    {
        MeshData b;
        const double h = track.width * 0.5 + 1;
        b.append(MeshBuilder::box(Vec3(0.15, 2.5, 0.15), Vec3(0.8, 0.8, 0.8), Vec3(-h, 2.5, 0)));
        b.append(MeshBuilder::box(Vec3(0.15, 2.5, 0.15), Vec3(0.8, 0.8, 0.8), Vec3(h, 2.5, 0)));
        for (int i = 0; i < 16; ++i)
        {
            const double x = -h + (i + 0.5) * (2 * h / 16);
            const Vec3 c   = (i % 2) ? Vec3(0.1, 0.1, 0.1) : Vec3(0.95, 0.95, 0.95);
            b.append(MeshBuilder::box(Vec3(h / 16, 0.4, 0.1), c, Vec3(x, 4.6, 0)));
        }
        const auto& g = track.checkpoints()[0];
        Vec3 p        = g.center;
        p(1)          = context->terrain(p(0), p(2));
        m.append(b, transformMatrix(p, yawRotation(headingOf(g.tangent))));
    }
    return m;
}

Entity RacingRender::focus()
{
    World& w = app->world();
    // The local player is a driver: its car
    const Entity mine = drivenVehicle(w, app->localPlayer());
    if (mine.valid() && w.has<VehicleState>(mine)) return mine;
    Entity first;
    w.view<const VehicleState>().each(
        [&](Entity e, const VehicleState&)
        {
            if (!first.valid()) first = e;
        });
    return first;
}

void RacingRender::beforeRender(double alpha)
{
    Vulkan::RenderScene& scene = *app->scene();
    worldRenderer.update();
    const double frameDt = clock.tick();
    World& w             = app->world();
    const Entity e       = focus();
    if (e != followed)
    {
        followed = e;
        camera.reset();
    }
    // Every car's wheels at its interpolated pose (the engine draws the body with the same pose), the suspension, spin
    // and steering between the last two (delayed) steps
    wheels.begin();
    w.view<const VehicleState>().each(
        [&](Entity car, const VehicleState& state)
        {
            Engine::WorldPose body;
            bool teleported = false;
            auto* c         = caches.entry(car);
            if (!c || !w.extract().interpolatedPose(car, alpha, body, &teleported)) return;
            if (teleported)
            {
                // Regression (CodeRabbit, S13b): a respawn between steps (the GUI button while paused) kept the old
                // wheels until the next step; reseed them from the VehicleState
                if (!readsRenderPose(w, car)) caches.reseed(car);
                if (car == followed) camera.reset();
            }
            const Wheels& a            = c->previous();
            const Wheels& b            = c->current();
            const VehicleParameters& p = w.data(state.model);
            const Engine::MeshId mesh  = w.findMesh(wheelMeshName(p.wheelRadius));
            if (!mesh) return;
            std::vector<Engine::MeshId> meshes(4, mesh);
            std::vector<Engine::WorldPose> poses(4, body);
            const double steer = a.steer + (b.steer - a.steer) * alpha;
            for (int i = 0; i < 4; ++i)
            {
                // Wheels at the suspension's current length, front wheels steered, spinning
                const double compression = a.compression[i] + (b.compression[i] - a.compression[i]) * alpha;
                const double spin        = a.spin[i] + (b.spin[i] - a.spin[i]) * alpha;
                const double length      = p.restLength + p.wheelRadius - compression;
                const Vec3 center        = p.wheelMounts[i] - Vec3(0, length - p.wheelRadius, 0);
                const Quat local  = (i < 2 ? yawRotation(-steer) : Quat::Identity()) * axisAngle(Vec3(1, 0, 0), -spin);
                poses[i].position = body.position + body.rotation * (center * body.scale);
                poses[i].rotation = (body.rotation * local).normalized();
            }
            wheels.set(car, meshes, poses);
            if (car == followed) camera.update(body.position, forwardOf(body.rotation), frameDt);
        });
    wheels.end();
    // While a tool holds the camera the app ignores it (S14)
    app->setViewCamera(0, camera.eye(), Vulkan::viewRotation(camera.view()),
                       radians(double(worldRenderer.settings.fovy)));
    // The time of the wind (the world's time between the steps)
    scene.setTime(w.time() + alpha * w.dt());
}

void RacingRender::gui()
{
    worldRenderer.imgui();
    // The GUI runs inside the frame (after beforeRender): the car it shows is focus()'s
    ImGui::SetNextWindowPos(ImVec2(10, ImGui::GetIO().DisplaySize.y - 170), ImGuiCond_FirstUseEver);
    ImGui::Begin("Racing", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Text("W/S throttle and brake, A/D steer, R back to the track, G GUI");
    World& w       = app->world();
    const Entity e = focus();
    if (w.alive(e) && w.has<VehicleState>(e))
    {
        const Vehicle car   = vehicleOf(w, e);
        const LapState& lap = w.read<LapState>(e);
        const bool mine     = drivenVehicle(w, app->localPlayer()) == e;
        const bool ai       = w.has<VehicleSeat>(e) && w.read<VehicleSeat>(e).driver == e;
        if (app->coop())
            ImGui::Text("co-op client (peer %d): %s, %d cars", int(w.localPeer()),
                        mine ? "your car" : (ai ? "the host's AI car" : "another player's car"),
                        int(w.registry().poolById(VehicleState::id)->size()));
        else
            ImGui::Text("%s", mine ? "player" : "demo (AI driver)");
        ImGui::Text("speed %.0f km/h, wheels on the ground %d", car.forwardSpeed() * 3.6, car.wheelsOnGround());
        ImGui::Text("lap %d, time %.2f s, last %.2f s, best %.2f s", lap.laps + 1, lap.lapTime, lap.lastLapTime,
                    lap.bestLapTime);
        ImGui::Text("next checkpoint %d / %d, track %.0f m", lap.nextCheckpoint,
                    int(context->track.checkpoints().size()), context->track.length());
        // The local player's action (a message on a co-op client; it acts when steps run); without a local player
        // (--demo on the host) the game object's respawn directly
        const Entity player = app->localPlayer();
        if (mine)
        {
            if (ImGui::Button(w.isPaused() ? "back to the track (when steps run)" : "back to the track"))
                w.queueAction(PlayerAction{player, uint32_t(RacingAction::Respawn), Vec3::Zero()});
        }
        else if (w.isHost() && !player.valid() && ImGui::Button("back to the track"))
        {
            if (auto* c = gameObjectAs<CarBase>(w, e)) c->respawn();
        }
    }
    ImGui::End();
}

}  // namespace Game
}  // namespace Saiga
