/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/app/EngineApp.h"
#include "saiga/vulkan/world/PrototypeScene.h"
#include "saiga/vulkan/world/WorldCommon.h"

#include "games/common/ChaseCamera.h"
#include "games/common/GameEngine.h"
#include "games/flight/FlightWorld.h"

namespace Saiga
{
namespace Game
{
/**
 * The render side of the flight demo (GameRenderHooks): the prototype environment (PrototypeScene without its forest:
 * sky, water, the foliage texture; S23c), the chase camera over the world's ground (flightGround), the GUI. The
 * terrain and the generated content (trees, rocks, ruins, grass) are the engine's. The airplanes are entities with
 * RenderInstance (drawn by the engine's extract, S19b; remote planes on a co-op client from RenderPose); the camera
 * follows the interpolated pose of the local player's plane (its driver's Driver.vehicle), or of the first airplane
 * (--demo).
 */
class FlightRender
{
   public:
    explicit FlightRender(std::shared_ptr<const FlightContext> context) : context(std::move(context)) {}

    void init(Engine::EngineApp& app);
    void shutdown();
    void beforeRender(double alpha);
    void gui();
    // The camera hold of a tool ended (EngineAppSetup::onCameraReleased)
    void cameraReleased() { camera.reset(); }

   private:
    Entity focus();

    std::shared_ptr<const FlightContext> context;
    Engine::EngineApp* app = nullptr;
    Vulkan::PrototypeScene prototype;
    ChaseCamera camera;
    FrameClock clock;
    Entity followed;
    uint64_t followedSnaps = 0;  // the followed plane's last snap seen (ExtractSystem::snapSequence)
    SpaceId cameraSpace    = 0;  // the followed plane's space (S21: the view camera's)
};

}  // namespace Game
}  // namespace Saiga
