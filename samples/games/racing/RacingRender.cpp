/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The render side of the racing demo: world prototype, track, car, chase camera, GUI.

#include "games/racing/RacingRender.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/engine/components/Player.h"
#include "saiga/vulkan/scene/RenderScene.h"

#include "games/common/WorldLighting.h"

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
    worldRenderer.init(app->scene()->forwardRenderer(), params,
                       std::shared_ptr<const Vulkan::HeightField>(context, &context->heightField));
    // Grass would grow through the track
    worldRenderer.settings.grass = false;
    meshes.init(app->scene()->forwardRenderer());

    const VehicleParameters vehicle;
    bodyMesh    = meshes.addMesh(MeshBuilder::carBody(Vec3(0.85, 0.15, 0.12)));
    wheelMesh   = meshes.addMesh(MeshBuilder::wheel(vehicle.wheelRadius, 0.28));
    coneMesh    = meshes.addMesh(MeshBuilder::trafficCone());
    trackMeshId = meshes.addMesh(trackMesh());
    {
        // Start / finish banner: two posts and a checkered bar
        MeshData b;
        double h = context->track.width * 0.5 + 1;
        b.append(MeshBuilder::box(Vec3(0.15, 2.5, 0.15), Vec3(0.8, 0.8, 0.8), Vec3(-h, 2.5, 0)));
        b.append(MeshBuilder::box(Vec3(0.15, 2.5, 0.15), Vec3(0.8, 0.8, 0.8), Vec3(h, 2.5, 0)));
        for (int i = 0; i < 16; ++i)
        {
            double x = -h + (i + 0.5) * (2 * h / 16);
            Vec3 c   = (i % 2) ? Vec3(0.1, 0.1, 0.1) : Vec3(0.95, 0.95, 0.95);
            b.append(MeshBuilder::box(Vec3(h / 16, 0.4, 0.1), c, Vec3(x, 4.6, 0)));
        }
        bannerMesh = meshes.addMesh(b);
    }

    const RacingContext* ctx = context.get();
    camera.distance          = 9;
    camera.height            = 3;
    camera.targetHeight      = 1.2;
    camera.stiffness         = 6;
    camera.ground            = [ctx](double x, double z) { return ctx->terrain(x, z); };
    pose.attach(app->world());
    // A load (load_snapshot, also paused) replaces the wheels: no interpolation from the state before
    loadListener = app->world().addLoadListener([this](const LoadInfo&) { reseedWheels(); });
    stepListener = app->world().addStepListener(
        [this]()
        {
            World& w       = app->world();
            const Entity e = pose.followed();
            if (!w.alive(e) || !w.has<VehicleState>(e)) return true;
            const VehicleState& s = w.read<VehicleState>(e);
            previousWheels        = currentWheels;
            currentWheels         = Wheels{s.compression, s.spin, s.steerAngle};
            return true;
        });
}

void RacingRender::reseedWheels()
{
    World& w       = app->world();
    const Entity e = pose.followed();
    if (!w.alive(e) || !w.has<VehicleState>(e)) return;
    const VehicleState& s = w.read<VehicleState>(e);
    currentWheels = previousWheels = Wheels{s.compression, s.spin, s.steerAngle};
}

void RacingRender::shutdown()
{
    pose.detach();
    app->world().removeStepListener(stepListener);
    app->world().removeLoadListener(loadListener);
    meshes.destroy();
    worldRenderer.destroy();
}

MeshData RacingRender::trackMesh() const
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
    return m;
}

Entity RacingRender::focus()
{
    World& w = app->world();
    Entity p = app->localPlayer();
    if (p.valid() && w.alive(p) && w.has<VehicleState>(p)) return p;
    Entity first;
    w.view<const VehicleState>().each(
        [&](Entity e, const VehicleState&)
        {
            if (!first.valid()) first = e;
        });
    return first;
}

void RacingRender::transfer(vk::CommandBuffer cmd)
{
    const double frameDt = clock.tick();
    const ivec2 size     = app->scene()->size();
    const float aspect   = float(size(0)) / float(std::max(1, size(1)));
    World& w             = app->world();
    const Entity e       = focus();
    if (e != pose.followed())
    {
        pose.follow(e);
        reseedWheels();
    }
    hasPose = pose.interpolated(w.alpha(), renderPose);
    if (pose.takeTeleport())
    {
        camera.reset();
        previousWheels = currentWheels;
    }
    const double alpha = w.alpha();
    for (int i = 0; i < 4; ++i)
    {
        renderWheels.compression[i] =
            previousWheels.compression[i] + (currentWheels.compression[i] - previousWheels.compression[i]) * alpha;
        renderWheels.spin[i] = previousWheels.spin[i] + (currentWheels.spin[i] - previousWheels.spin[i]) * alpha;
    }
    renderWheels.steer = previousWheels.steer + (currentWheels.steer - previousWheels.steer) * alpha;
    if (hasPose) camera.update(renderPose.position, forwardOf(renderPose.rotation), frameDt);

    const mat4 view = camera.view();
    app->camera().setView(view);
    worldRenderer.transfer(cmd, view, camera.eye().cast<float>(), aspect);
    meshes.transfer(cmd, view, camera.eye(), worldRenderer.projection(aspect), worldLighting(worldRenderer));
}

void RacingRender::render(vk::CommandBuffer cmd)
{
    const Track& track = context->track;
    worldRenderer.render(cmd);
    meshes.draw(trackMeshId, Mat4::Identity());
    // Cones at the gates, the banner at the start
    for (auto& g : track.checkpoints())
    {
        Vec3 r(-g.tangent(2), 0, g.tangent(0));
        for (double side : {-1.0, 1.0})
        {
            Vec3 p = g.center + r * (side * (track.width * 0.5 + 1.5));
            p(1)   = context->terrain(p(0), p(2));
            meshes.draw(coneMesh, transformMatrix(p, Quat::Identity()));
        }
    }
    {
        const auto& g = track.checkpoints()[0];
        Vec3 p        = g.center;
        p(1)          = context->terrain(p(0), p(2));
        meshes.draw(bannerMesh, transformMatrix(p, yawRotation(headingOf(g.tangent))));
    }

    World& w       = app->world();
    const Entity e = pose.followed();
    if (hasPose && w.alive(e) && w.has<VehicleState>(e))
    {
        const VehicleParameters& p = w.data(w.read<VehicleState>(e).model);
        Mat4 body                  = transformMatrix(renderPose.position, renderPose.rotation);
        meshes.draw(bodyMesh, body);
        for (int i = 0; i < 4; ++i)
        {
            // Wheels at the suspension's current length, front wheels steered, spinning
            Vec3 mount    = p.wheelMounts[i];
            double length = p.restLength + p.wheelRadius - renderWheels.compression[i];
            Vec3 center   = mount - Vec3(0, length - p.wheelRadius, 0);
            Quat steer    = i < 2 ? yawRotation(-renderWheels.steer) : Quat::Identity();
            Quat spin     = axisAngle(Vec3(1, 0, 0), -renderWheels.spin[i]);
            meshes.draw(wheelMesh, body * transformMatrix(center, steer * spin));
        }
    }
    meshes.render(cmd);
}

void RacingRender::gui()
{
    worldRenderer.imgui();
    // The GUI runs before transfer() in a frame: follow here too (the first frame's window had no fields)
    const Entity followed = focus();
    if (followed != pose.followed())
    {
        pose.follow(followed);
        reseedWheels();
    }
    ImGui::SetNextWindowPos(ImVec2(10, ImGui::GetIO().DisplaySize.y - 170), ImGuiCond_FirstUseEver);
    ImGui::Begin("Racing", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Text("W/S throttle and brake, A/D steer, R back to the track, G GUI");
    World& w       = app->world();
    const Entity e = pose.followed();
    if (w.alive(e) && w.has<VehicleState>(e))
    {
        const Vehicle car   = vehicleOf(w, e);
        const LapState& lap = w.read<LapState>(e);
        ImGui::Text("%s", app->localPlayer() == e ? "player" : "demo (AI driver)");
        ImGui::Text("speed %.0f km/h, wheels on the ground %d", car.forwardSpeed() * 3.6, car.wheelsOnGround());
        ImGui::Text("lap %d, time %.2f s, last %.2f s, best %.2f s", lap.laps + 1, lap.lapTime, lap.lastLapTime,
                    lap.bestLapTime);
        ImGui::Text("next checkpoint %d / %d, track %.0f m", lap.nextCheckpoint,
                    int(context->track.checkpoints().size()), context->track.length());
        if (ImGui::Button("back to the track") && w.has<Player>(e))
            w.queueAction(PlayerAction{e, uint32_t(RacingAction::Respawn), Vec3::Zero()});
    }
    ImGui::End();
}

}  // namespace Game
}  // namespace Saiga
