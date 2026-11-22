/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The render side of the platformer: level, animated characters (cosmetic instances), third-person camera, GUI.

#include "games/platformer/PlatformerRender.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/engine/components/Player.h"
#include "saiga/engine/gameplay/PlayerAction.h"
#include "saiga/engine/sync/VulkanRenderSink.h"
#include "saiga/vulkan/scene/RenderScene.h"
#include "saiga/vulkan/world/WorldCommon.h"

#include <algorithm>

namespace Saiga
{
namespace Game
{
using namespace Engine;

void PlatformerRender::init(EngineApp& _app)
{
    app = &_app;
    // The level of each space as one instance set in its space (S19b; S21: the cellar in space 1); the flag outdoors
    levels.resize(context->levels.size());
    for (size_t space = 0; space < context->levels.size(); ++space)
    {
        const Level& level = context->levels[space];
        MeshData lm;
        for (auto& b : level.boxes) lm.append(MeshBuilder::box((b.max - b.min) * 0.5, b.color, (b.max + b.min) * 0.5));
        if (space == 0)
            lm.append(MeshBuilder::flag(Vec3(0.9, 0.25, 0.2)), transformMatrix(level.goal, Quat::Identity()));
        const std::string name = space == 0 ? "platformer/level" : "platformer/level" + std::to_string(space);
        levels[space].create(*app->scene(), name, lm, space == 0 ? Vec3::Zero() : level.start, SpaceId(space));
    }
    parts.init(app->vulkanSink());
    for (int j = 0; j < JointCount; ++j) partMeshes.push_back(app->world().findMesh(heroPartMeshName(j)));

    camera.distance     = 7;
    camera.height       = 2.8;
    camera.targetHeight = 1.3;
    camera.stiffness    = 6;
    // Every character's motion; landings (the step that landed sets landingSpeed) as the pulse
    caches.attach(
        app->world(), [](World& w, const std::function<void(Entity)>& fn)
        { w.view<const CharacterMotion>().each([&](Entity e, const CharacterMotion&) { fn(e); }); },
        [](const World& w, Entity e)
        {
            const CharacterMotion& m = w.read<CharacterMotion>(e);
            return Motion{m.velocity, m.yaw, m.grounded, m.airTime, m.jumped, m.landingSpeed};
        },
        [](const Motion& m) { return m.landingSpeed; });
    // A load starts the animations over (no landing or blend from the state before)
    loadListener = app->world().addLoadListener([this](const LoadInfo&) { characters.clear(); });
}

void PlatformerRender::shutdown()
{
    caches.detach();
    app->world().removeLoadListener(loadListener);
    parts.clear();
    for (LevelGeometry& l : levels) l.release();
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

PlatformerRender::Character& PlatformerRender::character(Entity e)
{
    for (Character& c : characters)
        if (c.entity == e) return c;
    characters.push_back(Character());
    characters.back().entity = e;
    return characters.back();
}

void PlatformerRender::beforeRender(double alpha)
{
    const double frameDt = clock.tick();
    World& w             = app->world();
    const Entity e       = focus();
    bool snap            = false;
    if (e != followed)
    {
        followed = e;
        snap     = true;
    }
    // The animators of characters that are gone
    for (size_t i = 0; i < characters.size();)
        if (!w.alive(characters[i].entity))
            characters.erase(characters.begin() + std::ptrdiff_t(i));
        else
            characters[i++].drawn = false;

    // Every character: the interpolated pose (the extract's), the animation per frame from it and its (delayed) motion,
    // the body parts as cosmetic instances
    const bool toolCamera = app->cameraHeldByTool();
    Vec3 focusPosition    = Vec3::Zero();
    bool hasFocus         = false;
    parts.begin();
    w.view<const CharacterMotion>().each(
        [&](Entity ent, const CharacterMotion&)
        {
            auto* c = caches.entry(ent);
            Engine::WorldPose p;
            if (!c || !w.extract().interpolatedPose(ent, alpha, p)) return;
            const Motion& m = c->current();
            Character& ch   = character(ent);
            ch.position     = p.position;
            // The facing of the interpolated pose (with the position; also an edit of the rotation while paused)
            ch.yaw   = headingOf(forwardOf(p.rotation));
            ch.drawn = true;
            CharacterState s;
            s.position = p.position;
            s.velocity = m.velocity;
            s.yaw      = m.yaw;
            s.grounded = m.grounded;
            s.airTime  = m.airTime;
            s.jumped   = m.jumped;
            if (c->pending > 0) ch.animator.land(c->pending);
            c->pending = 0;
            ch.animator.update(s, Vec3(m.velocity(0), 0, m.velocity(2)).norm(), frameDt);
            const auto joints = jointTransforms(Skeleton::humanoid(), ch.animator.pose(), ch.position, ch.yaw);
            std::vector<Engine::WorldPose> poses(JointCount);
            for (int j = 0; j < JointCount; ++j)
            {
                poses[j].space    = p.space;
                poses[j].position = Vec3(joints[j](0, 3), joints[j](1, 3), joints[j](2, 3));
                Mat3 r;
                for (int a = 0; a < 3; ++a)
                    for (int b = 0; b < 3; ++b) r(a, b) = joints[j](a, b);
                poses[j].rotation = quat_cast(r);
            }
            if (std::find(partMeshes.begin(), partMeshes.end(), Engine::MeshId(0)) == partMeshes.end())
                parts.set(ent, partMeshes, poses);
            if (ent == followed)
            {
                focusPosition = p.position;
                // S21: the camera views the followed character's space (a change is a snap of its pose too)
                cameraSpace = p.space;
                hasFocus    = true;
                renderYaw   = ch.yaw;
                // A teleport (restart) snaps the camera once. Regression (S19b review): the level of the snap pair
                // reset the camera every frame while the hero stood still after its spawn, so Q/E did not turn it
                const uint64_t snaps = w.extract().snapSequence(ent);
                if (snaps != followedSnaps)
                {
                    followedSnaps = snaps;
                    snap          = true;
                }
                // The camera: Q/E for the player, behind the character in the demo (not while a tool holds it)
                if (!toolCamera && !snap)
                    cameraYaw += app->localPlayer() == ent ? app->input().axis("camera_turn") * 2.0 * frameDt
                                                           : wrapAngle(m.yaw - cameraYaw) * smoothFactor(1.5, frameDt);
            }
        });
    parts.end();
    if (snap)
    {
        camera.reset();
        cameraYaw = 0;
    }
    if (hasFocus) camera.update(focusPosition, forwardOf(yawRotation(cameraYaw)), frameDt);
    // While a tool holds the camera the app ignores it (S14)
    app->setViewCamera(cameraSpace, camera.eye(), Vulkan::viewRotation(camera.view()), radians(60.0));
}

void PlatformerRender::gui()
{
    // The GUI runs inside the frame (after beforeRender): the character it shows is focus()'s
    // Below the debug UI's bar
    ImGui::SetNextWindowPos(ImVec2(10, app->debugUi().barHeight() + 10), ImGuiCond_FirstUseEver);
    ImGui::Begin("Platformer", nullptr, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::Text("WASD move, shift walk, space jump, F door, Q/E camera, R restart, F1-F5 debug tabs, G GUI");
    World& w       = app->world();
    const Entity e = focus();
    if (w.alive(e) && w.has<CharacterMotion>(e))
    {
        const CharacterMotion& m = w.read<CharacterMotion>(e);
        // Regression (CodeRabbit, S13b): an entity with CharacterMotion but no RunState (made through the world API)
        // asserted here
        const RunState* run = w.has<RunState>(e) ? &w.read<RunState>(e) : nullptr;
        const bool mine     = app->localPlayer() == e;
        if (app->coop())
            ImGui::Text("co-op client (peer %d): %s, %d characters", int(w.localPeer()),
                        mine ? "your hero" : "another character",
                        int(w.registry().poolById(CharacterMotion::id)->size()));
        else
            ImGui::Text("%s", mine ? "player" : "demo");
        // S21: the space (F at a door: interact)
        const SpaceId space = w.read<Transform>(e).space();
        ImGui::Text("space %d '%s'", int(space), w.spaceTemplate(space).name.c_str());
        const CharacterAnimator& animator = character(e).animator;
        const auto& wt                    = animator.weights();
        ImGui::Text("blend weights: idle %.2f, move %.2f, air %.2f (run blend %.2f)", wt.idle, wt.move, wt.air,
                    animator.runBlend());
        ImGui::Text("speed %.1f m/s, %s", Vec3(m.velocity(0), 0, m.velocity(2)).norm(),
                    m.grounded ? "grounded" : "in the air");
        if (run && run->goalTime >= 0)
            ImGui::Text("goal reached in %.1f s", run->goalTime);
        else if (run)
            ImGui::Text("time %.1f s", run->runTime);
        // The local player's action (a message on a co-op client; it acts when steps run); without a local player
        // (--demo on the host) the game object's restart directly
        const Entity player = app->localPlayer();
        if (mine)
        {
            if (ImGui::Button(w.isPaused() ? "restart (when steps run)" : "restart"))
                w.queueAction(PlayerAction{player, uint32_t(PlatformerAction::Restart), Vec3::Zero()});
        }
        else if (w.isHost() && !player.valid() && ImGui::Button("restart"))
        {
            // Between steps, not while the frame is recorded (S19c)
            app->queueBetweenSteps(
                [this, e]()
                {
                    World& world = app->world();
                    if (!world.alive(e)) return;
                    if (auto* hero = gameObjectAs<HeroBase>(world, e))
                    {
                        hero->restart();
                        // Between steps (paused) the cache would keep the motion from before the restart
                        caches.reseed(e);
                    }
                });
        }
    }
    ImGui::End();
}

}  // namespace Game
}  // namespace Saiga
