/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/app/EngineApp.h"
#include "saiga/vulkan/world/PrototypeScene.h"

#include "games/common/ChaseCamera.h"
#include "games/common/CosmeticDynamics.h"
#include "games/common/GameEngine.h"
#include "games/common/RenderAssets.h"
#include "games/racing/RacingWorld.h"

namespace Saiga
{
namespace Game
{
/**
 * The render side of the racing demo (GameRenderHooks): the world prototype on the context's height field (no grass: it
 * would grow through the track), the track with its cones and banner (constant level geometry, one instance set), the
 * wheels of every car (cosmetic dynamic instances at the car's interpolated pose; their compression, spin and steering
 * between the last two steps, read as late as the body on a co-op client, S16b), the chase camera, the GUI. The car
 * bodies are entities with RenderInstance (drawn by the engine's extract, S19b). Follows the local player's car (its
 * driver's Driver.vehicle), or the first car (--demo).
 */
class RacingRender
{
   public:
    explicit RacingRender(std::shared_ptr<const RacingContext> context) : context(std::move(context)) {}

    void init(Engine::EngineApp& app);
    void shutdown();
    void beforeRender(double alpha);
    void gui();
    // The camera hold of a tool ended (EngineAppSetup::onCameraReleased)
    void cameraReleased() { camera.reset(); }

   private:
    struct Wheels
    {
        std::array<double, 4> compression = {}, spin = {};
        double steer = 0;
    };
    Entity focus();
    MeshData levelMesh() const;

    std::shared_ptr<const RacingContext> context;
    Engine::EngineApp* app = nullptr;
    Vulkan::PrototypeScene prototype;
    LevelGeometry level;
    CosmeticDynamics wheels;
    ChaseCamera camera;
    RenderCaches<Wheels> caches;
    FrameClock clock;
    Entity followed;
};

}  // namespace Game
}  // namespace Saiga
