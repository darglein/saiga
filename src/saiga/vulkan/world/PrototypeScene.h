/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vulkan/world/ForestRenderer.h"
#include "saiga/vulkan/world/PrototypeForest.h"
#include "saiga/vulkan/world/WorldSettings.h"

#include <vector>

namespace Saiga
{
namespace Vulkan
{
class RenderScene;

/**
 * The game side of the world prototype on a RenderScene (S20, replacing WorldRenderer; docs/engine_design.md §19, A15):
 * the settings (WorldSettings: GUI, game_* MCP tools) with their mapping to RenderScene's PrototypeSettings, and the
 * trees (PrototypeForest). The terrain, grass and water come from the engine (the World's terrain through the render
 * sink); the environment (sky, sun, fog, the foliage texture) from setPrototypeEnvironment. The frame and terrain
 * statistics are in the app's "Engine statistics" (F1); the window shows the tree counts.
 *
 *   PrototypeScene prototype;
 *   prototype.attach(scene, trees, seed);   // renderHooks.init: setPrototypeEnvironment and the trees' instance sets
 *   per frame before the scene records it:   // e.g. beforeRender
 *       prototype.update(); scene.setTime(world time);
 */
class SAIGA_VULKAN_API PrototypeScene
{
   public:
    PrototypeScene() = default;
    // It owns the trees' instance set handles (a copy would remove them twice)
    PrototypeScene(const PrototypeScene&)            = delete;
    PrototypeScene& operator=(const PrototypeScene&) = delete;
    ~PrototypeScene() { detach(); }

    // setPrototypeEnvironment(settings, seed) and the instance sets of the placed trees (outside of renderFrame; seed:
    // the foliage texture and the tree species). A second attach (also to another scene) detaches first; the tree
    // meshes stay registered on a scene and are reused by name
    void attach(RenderScene& scene, const std::vector<ForestRenderer::TreeInstance>& trees, uint32_t seed);
    // Removes the instance sets (if the scene lives on) and forgets the scene
    void detach();
    // Per frame, before the scene records the frame: the settings (setPrototypeSettings, the trees)
    void update();
    // The settings window with the tree counts
    void imgui();

    WorldSettings settings;
    PrototypeForest& forest() { return trees; }
    const PrototypeForest& forest() const { return trees; }
    RenderScene* scene() const { return renderScene; }

   private:
    RenderScene* renderScene = nullptr;
    PrototypeForest trees;
};

}  // namespace Vulkan
}  // namespace Saiga
