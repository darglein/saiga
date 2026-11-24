/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/json11.hpp"
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
 * sink); the foliage texture from setFoliageTexture, the light from the environment (S36). The frame and terrain
 * statistics are in the engine's debug UI; imgui() shows the settings and the tree counts.
 *
 *   PrototypeScene prototype;
 *   prototype.attach(scene, trees, seed);   // renderHooks.init: setFoliageTexture and the trees' instance sets
 *   per frame before the scene records it:   // e.g. beforeRender
 *       prototype.update();
 */
class SAIGA_VULKAN_API PrototypeScene
{
   public:
    PrototypeScene() = default;
    // It owns the trees' instance set handles (a copy would remove them twice)
    PrototypeScene(const PrototypeScene&)            = delete;
    PrototypeScene& operator=(const PrototypeScene&) = delete;
    ~PrototypeScene() { detach(); }

    // setFoliageTexture(seed), the settings and the instance sets of the placed trees (outside of renderFrame; seed:
    // the foliage texture and the tree species). A second attach (also to another scene) detaches first; the tree
    // meshes stay registered on a scene and are reused by name
    void attach(RenderScene& scene, const std::vector<ForestRenderer::TreeInstance>& trees, uint32_t seed);
    // Removes the instance sets (if the scene lives on) and forgets the scene
    void detach();
    // Per frame, before the scene records the frame: the settings (setPrototypeSettings, the trees)
    void update();
    // The settings with the tree counts: widgets only (no window; a section of the engine's debug UI, S19c)
    void imgui();
    // The render statistics of the newest frames (game_get_state): GPU frame and sections, terrain nodes and triangles,
    // trees per LOD, grass, draw calls, MSAA, vsync, GPU memory of the terrain set and the trees
    json11::Json statsJson() const;

    WorldSettings settings;
    PrototypeForest& forest() { return trees; }
    const PrototypeForest& forest() const { return trees; }
    RenderScene* scene() const { return renderScene; }

   private:
    RenderScene* renderScene = nullptr;
    mutable PrototypeForest trees;
};

}  // namespace Vulkan
}  // namespace Saiga
