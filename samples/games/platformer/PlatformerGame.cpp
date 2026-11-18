/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The platformer tech demo: level, animated character, third-person camera.

#include "games/platformer/PlatformerGame.h"

#include "saiga/core/imgui/imgui.h"

namespace Saiga
{
namespace Game
{
void PlatformerGame::init(Vulkan::VulkanForwardRenderer& _renderer, bool _demo)
{
    renderer = &_renderer;
    demo     = _demo;
    meshes.init(*renderer);
    level = Level::demoLevel();

    // The level as one mesh
    MeshData lm;
    for (auto& b : level.boxes) lm.append(MeshBuilder::box((b.max - b.min) * 0.5, b.color, (b.max + b.min) * 0.5));
    levelMesh = meshes.addMesh(lm);
    flagMesh  = meshes.addMesh(MeshBuilder::flag(Vec3(0.9, 0.25, 0.2)));

    // Body parts: boxes hanging from their joints
    Vec3 skin(0.95, 0.75, 0.6), shirt(0.25, 0.45, 0.8), pants(0.3, 0.3, 0.35), shoes(0.2, 0.15, 0.12);
    auto part = [&](int joint, const Vec3& half, const Vec3& center, const Vec3& color)
    {
        partMesh[joint]   = meshes.addMesh(MeshBuilder::box(half, color, center));
        partOffset[joint] = Mat4::Identity();
    };
    part(JointPelvis, Vec3(0.18, 0.1, 0.11), Vec3(0, 0, 0), pants);
    part(JointSpine, Vec3(0.22, 0.25, 0.13), Vec3(0, 0.27, 0), shirt);
    part(JointHead, Vec3(0.14, 0.15, 0.14), Vec3(0, 0.15, 0), skin);
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
        partMesh[JointShinL]   = meshes.addMesh(shin);
        partMesh[JointShinR]   = meshes.addMesh(shin);
        partOffset[JointShinL] = partOffset[JointShinR] = Mat4::Identity();
    }
    // Eyes, so the facing is visible
    {
        MeshData head = MeshBuilder::box(Vec3(0.14, 0.15, 0.14), skin, Vec3(0, 0.15, 0));
        head.append(MeshBuilder::box(Vec3(0.025, 0.03, 0.01), Vec3(0.1, 0.1, 0.12), Vec3(-0.06, 0.18, -0.14)));
        head.append(MeshBuilder::box(Vec3(0.025, 0.03, 0.01), Vec3(0.1, 0.1, 0.12), Vec3(0.06, 0.18, -0.14)));
        partMesh[JointHead] = meshes.addMesh(head);
    }

    character.level     = &level;
    camera.distance     = 7;
    camera.height       = 2.8;
    camera.targetHeight = 1.3;
    camera.stiffness    = 6;
    respawn();

    if (demo)
    {
        // Run a while before the first frame (SAIGA_SCREENSHOT then shows the character running over the bridge)
        for (int i = 0; i < int(60 * 4.6); ++i)
        {
            step(PlayerInput(), 1.0 / 60);
            animate(character.state, 1.0 / 60);
        }
    }
}

void PlatformerGame::destroy()
{
    meshes.destroy();
}

void PlatformerGame::respawn()
{
    character.reset(level.start, 0);
    previous   = character.state;
    demoRunner = PlatformerDemo();
    cameraYaw  = 0;
    goalTime   = -1;
    runTime    = 0;
    // The camera jumps along in the next frame (it would glide from the goal or the fall back to the start otherwise)
    camera.reset();
}

void PlatformerGame::step(const PlayerInput& in, double dt)
{
    previous = character.state;
    runTime += dt;
    CharacterInput ci;
    if (demo)
    {
        ci = demoRunner.input(character, level);
        // The camera turns behind the character
        cameraYaw += wrapAngle(character.state.yaw - cameraYaw) * smoothFactor(1.5, dt);
    }
    else
    {
        cameraYaw += in.cameraTurn * dt;
        Vec3 f = forwardOf(yawRotation(cameraYaw)), r = rightOf(yawRotation(cameraYaw));
        ci.move        = f * in.forward + r * in.right;
        ci.walk        = in.walk;
        ci.jumpHeld    = in.jumpHeld;
        ci.jumpPressed = in.jumpPressed;
    }
    character.step(ci, dt);
    if (character.state.landingSpeed > 0) pendingLanding = std::max(pendingLanding, character.state.landingSpeed);
    if (character.state.position(1) < level.killHeight) respawn();
    if (goalTime < 0 && demoRunner.reachedGoal(character, level)) goalTime = runTime;
    // The demo starts over a few seconds after reaching the goal
    if (demo && goalTime >= 0 && runTime > goalTime + 3) respawn();
}

void PlatformerGame::transfer(vk::CommandBuffer cmd, float aspect, double frameDt, double alpha)
{
    aspectRatio   = aspect;
    const auto& s = character.state;
    renderFeet    = previous.position + (s.position - previous.position) * alpha;
    renderYaw     = previous.yaw + wrapAngle(s.yaw - previous.yaw) * alpha;

    // The animation runs per frame from the interpolated state
    CharacterState interpolated = s;
    interpolated.position       = renderFeet;
    interpolated.velocity       = previous.velocity + (s.velocity - previous.velocity) * alpha;
    animate(interpolated, frameDt);

    camera.update(renderFeet, forwardOf(yawRotation(cameraYaw)), frameDt);
    mat4 view = camera.view();
    MeshRenderer::Lighting light;
    light.sunDirection = Vec3(0.4, 0.8, 0.35);
    light.ambient      = 0.35;
    meshes.transfer(cmd, view, camera.eye(), Vulkan::reversedZInfinitePerspective(radians(60.0f), aspect, 0.1f), light);
}

void PlatformerGame::animate(const CharacterState& state, double dt)
{
    if (pendingLanding > 0) animator.land(pendingLanding);
    pendingLanding = 0;
    Vec3 hv(state.velocity(0), 0, state.velocity(2));
    animator.update(state, hv.norm(), dt);
}

void PlatformerGame::render(vk::CommandBuffer cmd)
{
    meshes.draw(levelMesh, Mat4::Identity());
    meshes.draw(flagMesh, transformMatrix(level.goal, Quat::Identity()));
    auto joints = jointTransforms(Skeleton::humanoid(), animator.pose(), renderFeet, renderYaw);
    for (int j = 0; j < JointCount; ++j) meshes.draw(partMesh[j], joints[j] * partOffset[j]);
    meshes.render(cmd);
}

void PlatformerGame::imgui()
{
    ImGui::SetNextWindowPos(ImVec2(10, 10), ImGuiCond_FirstUseEver);
    ImGui::Begin("Platformer");
    ImGui::Text("WASD move, shift walk, space jump, Q/E camera, R restart, G GUI");
    ImGui::Checkbox("demo", &demo);
    const auto& w = animator.weights();
    ImGui::Text("blend weights: idle %.2f, move %.2f, air %.2f (run blend %.2f)", w.idle, w.move, w.air,
                animator.runBlend());
    ImGui::Text("speed %.1f m/s, %s", character.horizontalSpeed(),
                character.state.grounded ? "grounded" : "in the air");
    if (goalTime >= 0)
        ImGui::Text("goal reached in %.1f s", goalTime);
    else
        ImGui::Text("time %.1f s", runTime);
    if (ImGui::Button("restart")) respawn();
    ImGui::End();
}

}  // namespace Game
}  // namespace Saiga
