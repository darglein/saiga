/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The render side of the platformer: level, animated character, third-person camera, GUI.

#include "games/platformer/PlatformerRender.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/engine/components/Player.h"
#include "saiga/vulkan/scene/RenderScene.h"

namespace Saiga
{
namespace Game
{
using namespace Engine;

void PlatformerRender::init(EngineApp& _app)
{
    app = &_app;
    meshes.init(app->scene()->forwardRenderer());
    const Level& level = context->level;

    // The level as one mesh
    MeshData lm;
    for (auto& b : level.boxes) lm.append(MeshBuilder::box((b.max - b.min) * 0.5, b.color, (b.max + b.min) * 0.5));
    levelMesh = meshes.addMesh(lm);
    flagMesh  = meshes.addMesh(MeshBuilder::flag(Vec3(0.9, 0.25, 0.2)));

    // Body parts: boxes hanging from their joints
    Vec3 skin(0.95, 0.75, 0.6), shirt(0.25, 0.45, 0.8), pants(0.3, 0.3, 0.35), shoes(0.2, 0.15, 0.12);
    auto part = [&](int joint, const Vec3& half, const Vec3& center, const Vec3& color)
    { partMesh[joint] = meshes.addMesh(MeshBuilder::box(half, color, center)); };
    part(JointPelvis, Vec3(0.18, 0.1, 0.11), Vec3(0, 0, 0), pants);
    part(JointSpine, Vec3(0.22, 0.25, 0.13), Vec3(0, 0.27, 0), shirt);
    part(JointUpperArmL, Vec3(0.065, 0.15, 0.065), Vec3(0, -0.14, 0), shirt);
    part(JointUpperArmR, Vec3(0.065, 0.15, 0.065), Vec3(0, -0.14, 0), shirt);
    part(JointLowerArmL, Vec3(0.055, 0.15, 0.055), Vec3(0, -0.15, 0), skin);
    part(JointLowerArmR, Vec3(0.055, 0.15, 0.055), Vec3(0, -0.15, 0), skin);
    part(JointThighL, Vec3(0.085, 0.22, 0.085), Vec3(0, -0.22, 0), pants);
    part(JointThighR, Vec3(0.085, 0.22, 0.085), Vec3(0, -0.22, 0), pants);
    {
        // Shins with feet
        MeshData shin = MeshBuilder::box(Vec3(0.075, 0.2, 0.075), pants, Vec3(0, -0.2, 0));
        shin.append(MeshBuilder::box(Vec3(0.08, 0.05, 0.13), shoes, Vec3(0, -0.4, -0.05)));
        partMesh[JointShinL] = meshes.addMesh(shin);
        partMesh[JointShinR] = meshes.addMesh(shin);
    }
    {
        // The head with eyes, so the facing is visible
        MeshData head = MeshBuilder::box(Vec3(0.14, 0.15, 0.14), skin, Vec3(0, 0.15, 0));
        head.append(MeshBuilder::box(Vec3(0.025, 0.03, 0.01), Vec3(0.1, 0.1, 0.12), Vec3(-0.06, 0.18, -0.14)));
        head.append(MeshBuilder::box(Vec3(0.025, 0.03, 0.01), Vec3(0.1, 0.1, 0.12), Vec3(0.06, 0.18, -0.14)));
        partMesh[JointHead] = meshes.addMesh(head);
    }

    camera.distance     = 7;
    camera.height       = 2.8;
    camera.targetHeight = 1.3;
    camera.stiffness    = 6;
    pose.attach(app->world());
    // Landings for the animation (the step that landed sets landingSpeed)
    app->world().addStepListener(
        [this]()
        {
            World& w       = app->world();
            const Entity e = pose.followed();
            if (w.alive(e) && w.has<CharacterMotion>(e))
                pendingLanding = std::max(pendingLanding, w.read<CharacterMotion>(e).landingSpeed);
            return true;
        });
}

void PlatformerRender::shutdown()
{
    meshes.destroy();
}

Entity PlatformerRender::focus()
{
    World& w = app->world();
    Entity p = app->localPlayer();
    if (p.valid() && w.alive(p) && w.has<CharacterMotion>(p)) return p;
    Entity first;
    w.view<const CharacterMotion>().each(
        [&](Entity e, const CharacterMotion&)
        {
            if (!first.valid()) first = e;
        });
    return first;
}

void PlatformerRender::transfer(vk::CommandBuffer cmd)
{
    const double frameDt = clock.tick();
    const ivec2 size     = app->scene()->size();
    const float aspect   = float(size(0)) / float(std::max(1, size(1)));
    World& w             = app->world();
    const Entity e       = focus();
    pose.follow(e);
    hasPose = pose.interpolated(w.alpha(), renderPose);
    if (pose.takeTeleport())
    {
        camera.reset();
        cameraYaw = 0;
    }
    if (hasPose)
    {
        const CharacterMotion& m = w.read<CharacterMotion>(e);
        renderYaw                = m.yaw;
        // The camera: Q/E for the player, behind the character in the demo
        if (app->localPlayer() == e)
            cameraYaw += app->input().axis("camera_turn") * 2.0 * frameDt;
        else
            cameraYaw += wrapAngle(m.yaw - cameraYaw) * smoothFactor(1.5, frameDt);

        // The animation runs per frame from the interpolated pose
        CharacterState s;
        s.position = renderPose.position;
        s.velocity = m.velocity;
        s.yaw      = m.yaw;
        s.grounded = m.grounded;
        s.airTime  = m.airTime;
        s.jumped   = m.jumped;
        if (pendingLanding > 0) animator.land(pendingLanding);
        pendingLanding = 0;
        animator.update(s, Vec3(m.velocity(0), 0, m.velocity(2)).norm(), frameDt);
        camera.update(renderPose.position, forwardOf(yawRotation(cameraYaw)), frameDt);
    }
    const mat4 view = camera.view();
    app->camera().setView(view);
    MeshRenderer::Lighting light;
    light.sunDirection = Vec3(0.4, 0.8, 0.35);
    light.ambient      = 0.35;
    meshes.transfer(cmd, view, camera.eye(), Vulkan::reversedZInfinitePerspective(radians(60.0f), aspect, 0.1f), light);
}

void PlatformerRender::render(vk::CommandBuffer cmd)
{
    meshes.draw(levelMesh, Mat4::Identity());
    meshes.draw(flagMesh, transformMatrix(context->level.goal, Quat::Identity()));
    if (hasPose)
    {
        auto joints = jointTransforms(Skeleton::humanoid(), animator.pose(), renderPose.position, renderYaw);
        for (int j = 0; j < JointCount; ++j) meshes.draw(partMesh[j], joints[j]);
    }
    meshes.render(cmd);
}

void PlatformerRender::gui()
{
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::Begin("Platformer");
    ImGui::Text("WASD move, shift walk, space jump, Q/E camera, R restart, G GUI");
    World& w       = app->world();
    const Entity e = pose.followed();
    if (w.alive(e) && w.has<CharacterMotion>(e))
    {
        const CharacterMotion& m = w.read<CharacterMotion>(e);
        const RunState& run      = w.read<RunState>(e);
        ImGui::Text("%s", app->localPlayer() == e ? "player" : "demo");
        const auto& wt = animator.weights();
        ImGui::Text("blend weights: idle %.2f, move %.2f, air %.2f (run blend %.2f)", wt.idle, wt.move, wt.air,
                    animator.runBlend());
        ImGui::Text("speed %.1f m/s, %s", Vec3(m.velocity(0), 0, m.velocity(2)).norm(),
                    m.grounded ? "grounded" : "in the air");
        if (run.goalTime >= 0)
            ImGui::Text("goal reached in %.1f s", run.goalTime);
        else
            ImGui::Text("time %.1f s", run.runTime);
        if (ImGui::Button("restart") && w.has<Player>(e))
            w.queueAction(PlayerAction{e, uint32_t(PlatformerAction::Restart), Vec3::Zero()});
    }
    ImGui::End();
}

}  // namespace Game
}  // namespace Saiga
