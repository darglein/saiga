/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/app/EngineApp.h"

#include "games/common/ChaseCamera.h"
#include "games/common/CosmeticDynamics.h"
#include "games/common/GameEngine.h"
#include "games/common/RenderAssets.h"
#include "games/platformer/CharacterAnimation.h"
#include "games/platformer/PlatformerWorld.h"

#include <array>

namespace Saiga
{
namespace Game
{
/**
 * The render side of the platformer (GameRenderHooks): the level and the flag (constant level geometry, one instance
 * set), every humanoid built from boxes with the procedural animation (cosmetic dynamic instances at the character's
 * interpolated pose from the engine's extract, S19b; render-side until S34; one animator per character, landings come
 * from the step listener of the render caches; S16b: remote characters on a co-op client from RenderPose, their motion
 * read as late), the third-person camera (Q/E turn it; the demo's turns behind the character), the GUI. The heroes
 * are tracked by the extract (an empty RenderInstance), not drawn by it.
 */
class PlatformerRender
{
   public:
    explicit PlatformerRender(std::shared_ptr<const PlatformerContext> context) : context(std::move(context)) {}

    void init(Engine::EngineApp& app);
    void shutdown();
    void beforeRender(double alpha);
    void gui();
    // The camera hold of a tool ended (EngineAppSetup::onCameraReleased): back behind the character
    void cameraReleased()
    {
        camera.reset();
        cameraYaw = renderYaw;
    }

   private:
    // The motion the animation reads (CharacterMotion of a step)
    struct Motion
    {
        Vec3 velocity       = Vec3::Zero();
        double yaw          = 0;
        bool grounded       = false;
        double airTime      = 0;
        bool jumped         = false;
        double landingSpeed = 0;
    };
    // A character as drawn this frame
    struct Character
    {
        Entity entity;
        CharacterAnimator animator;
        Vec3 position = Vec3::Zero();
        double yaw    = 0;
        bool drawn    = false;
    };
    Entity focus();
    Character& character(Entity e);

    std::shared_ptr<const PlatformerContext> context;
    Engine::EngineApp* app = nullptr;
    // The level geometry per space (S21: the outdoor level with the flag, the cellar)
    std::vector<LevelGeometry> levels;
    SpaceId cameraSpace = 0;  // the followed character's space (the view camera's)
    CosmeticDynamics parts;
    ChaseCamera camera;
    RenderCaches<Motion> caches;
    std::vector<Character> characters;
    FrameClock clock;
    Entity followed;
    uint64_t followedSnaps = 0;  // the followed character's last snap seen (ExtractSystem::snapSequence)
    double cameraYaw       = 0;
    double renderYaw       = 0;  // the followed character's
    int loadListener       = -1;
    std::vector<Engine::MeshId> partMeshes;
};

}  // namespace Game
}  // namespace Saiga
