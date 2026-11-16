/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The racing tech demo: track on the world prototype, car, laps, chase camera.

#include "games/racing/RacingGame.h"

#include "saiga/core/imgui/imgui.h"

#include "games/common/WorldLighting.h"

namespace Saiga
{
namespace Game
{
void RacingGame::init(Vulkan::VulkanForwardRenderer& renderer, bool _demo, int treeCount)
{
    demo = _demo;
    Vulkan::WorldParameters params;
    params.treeCount = treeCount;
    worldRenderer.init(renderer, params);
    // Grass would grow through the track
    worldRenderer.settings.grass = false;
    meshes.init(renderer);

    GroundHeight ground = worldGround(worldRenderer, false);
    const auto& hf      = worldRenderer.heightField();
    // A site on gentle land with few trees on the track (the prototype's trees cannot be removed): count the trees
    // per 16 m cell
    const double cell = 16;
    int cells         = int(hf.worldSize() / cell) + 1;
    std::vector<int> trees(size_t(cells) * cells, 0);
    for (auto& t : worldRenderer.forest().treeInstances())
    {
        int cx = int(t.posYaw(0) / cell), cz = int(t.posYaw(2) / cell);
        if (cx >= 0 && cz >= 0 && cx < cells && cz < cells) trees[size_t(cz) * cells + cx]++;
    }
    auto treePenalty = [&](const Vec3& p)
    {
        int cx = int(p(0) / cell), cz = int(p(2) / cell);
        return (cx >= 0 && cz >= 0 && cx < cells && cz < cells) ? 3.0 * trees[size_t(cz) * cells + cx] : 0.0;
    };
    const double radius = 220;
    const uint32_t seed = 7;
    Vec3 site           = Track::findSite(ground, hf.worldSize(), radius, hf.waterLevel() + 3, 12, seed, treePenalty);
    track.width         = 12;
    track.build(Track::loop(site, radius, 12, seed), 10);

    terrain = ground;
    // The car drives on the track surface (lifted asphalt) that is drawn, the terrain elsewhere
    car.ground = [this](double x, double z) { return terrain(x, z) + track.surfaceOffset(Vec3(x, 0, z), surfaceHint); };
    bodyMesh   = meshes.addMesh(MeshBuilder::carBody(Vec3(0.85, 0.15, 0.12)));
    wheelMesh  = meshes.addMesh(MeshBuilder::wheel(car.params.wheelRadius, 0.28));
    coneMesh   = meshes.addMesh(MeshBuilder::trafficCone());
    trackMeshId = meshes.addMesh(trackMesh(ground));
    {
        // Start / finish banner: two posts and a checkered bar
        MeshData b;
        double h = track.width * 0.5 + 1;
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

    camera.distance     = 9;
    camera.height       = 3;
    camera.targetHeight = 1.2;
    camera.stiffness    = 6;
    camera.ground       = ground;

    Vec3 start = track.position(0);
    car.reset(Vec3(start(0), std::nan(""), start(2)), headingOf(track.tangent(0)));
    previous = car.state;
    timer.reset(track);

    if (demo)
    {
        // Race a while before the first frame (SAIGA_SCREENSHOT shows the car on the track)
        for (int i = 0; i < 60 * 10; ++i) step(VehicleControls(), 1.0 / 60);
    }
}

void RacingGame::destroy()
{
    meshes.destroy();
    worldRenderer.destroy();
}

MeshData RacingGame::trackMesh(const GroundHeight& ground) const
{
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
        p(1)   = ground(p(0), p(2)) + lift;
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

void RacingGame::respawn()
{
    const auto& gates = track.checkpoints();
    int last          = (timer.nextCheckpoint - 1 + int(gates.size())) % int(gates.size());
    Vec3 p            = track.position(gates[last].s + 3);
    // Regression: the track search starts at the hint; a stale hint from the crash site missed the asphalt lift and the
    // car spawned 0.15 m too low (it hopped)
    surfaceHint = -1;
    car.reset(Vec3(p(0), std::nan(""), p(2)), headingOf(track.tangent(gates[last].s + 3)));
    previous        = car.state;
    driver.progress = -1;
    upsideDown      = 0;
    // The camera jumps along in the next frame (it would glide from the crash site otherwise)
    camera.reset();
}

void RacingGame::step(const VehicleControls& player, double dt)
{
    previous     = car.state;
    lastControls = demo ? driver.controls(car, track) : player;
    Vec3 before  = car.state.position;
    car.step(lastControls, dt);
    timer.update(track, before, car.state.position, dt);
    // Stuck on the roof: back on the track
    upsideDown = car.up()(1) < 0.3 ? upsideDown + dt : 0;
    if (upsideDown > 2.0) respawn();
}

void RacingGame::transfer(vk::CommandBuffer cmd, float aspect, double frameDt, double alpha)
{
    renderPosition = previous.position + (car.state.position - previous.position) * alpha;
    renderRotation = previous.orientation.slerp(alpha, car.state.orientation);
    for (int i = 0; i < 4; ++i)
    {
        renderCompression[i] = previous.compression[i] + (car.state.compression[i] - previous.compression[i]) * alpha;
        renderSpin[i]        = previous.spin[i] + (car.state.spin[i] - previous.spin[i]) * alpha;
    }
    renderSteer = previous.steerAngle + (car.state.steerAngle - previous.steerAngle) * alpha;
    camera.update(renderPosition, forwardOf(renderRotation), frameDt);

    mat4 view = camera.view();
    worldRenderer.transfer(cmd, view, camera.eye().cast<float>(), aspect);
    meshes.transfer(cmd, view, camera.eye(), worldRenderer.projection(aspect), worldLighting(worldRenderer));
}

void RacingGame::render(vk::CommandBuffer cmd)
{
    worldRenderer.render(cmd);
    meshes.draw(trackMeshId, Mat4::Identity());
    // Cones at the gates, the banner at the start
    for (auto& g : track.checkpoints())
    {
        Vec3 r(-g.tangent(2), 0, g.tangent(0));
        for (double side : {-1.0, 1.0})
        {
            Vec3 p = g.center + r * (side * (track.width * 0.5 + 1.5));
            p(1)   = terrain(p(0), p(2));
            meshes.draw(coneMesh, transformMatrix(p, Quat::Identity()));
        }
    }
    {
        const auto& g = track.checkpoints()[0];
        Vec3 p        = g.center;
        p(1)          = terrain(p(0), p(2));
        meshes.draw(bannerMesh, transformMatrix(p, yawRotation(headingOf(g.tangent))));
    }

    Mat4 body = transformMatrix(renderPosition, renderRotation);
    meshes.draw(bodyMesh, body);
    for (int i = 0; i < 4; ++i)
    {
        // Wheels at the suspension's current length, front wheels steered, spinning
        const auto& p = car.params;
        Vec3 mount    = p.wheelMounts[i];
        double length = p.restLength + p.wheelRadius - renderCompression[i];
        Vec3 center   = mount - Vec3(0, length - p.wheelRadius, 0);
        Quat steer    = i < 2 ? yawRotation(-renderSteer) : Quat::Identity();
        Quat spin     = axisAngle(Vec3(1, 0, 0), -renderSpin[i]);
        meshes.draw(wheelMesh, body * transformMatrix(center, steer * spin));
    }
    meshes.render(cmd);
}

void RacingGame::imgui()
{
    worldRenderer.imgui();
    ImGui::SetNextWindowPos(ImVec2(10, ImGui::GetIO().DisplaySize.y - 170), ImGuiCond_FirstUseEver);
    ImGui::Begin("Racing");
    ImGui::Text("W/S throttle and brake, A/D steer, R back to the track, G GUI");
    ImGui::Checkbox("demo (AI driver)", &demo);
    ImGui::Text("speed %.0f km/h, wheels on the ground %d", car.forwardSpeed() * 3.6, car.wheelsOnGround());
    ImGui::Text("lap %d, time %.2f s, last %.2f s, best %.2f s", timer.laps + 1, timer.lapTime, timer.lastLapTime,
                timer.bestLapTime);
    ImGui::Text("next checkpoint %d / %d, track %.0f m", timer.nextCheckpoint, int(track.checkpoints().size()),
                track.length());
    if (ImGui::Button("back to the track")) respawn();
    ImGui::End();
}

}  // namespace Game
}  // namespace Saiga
