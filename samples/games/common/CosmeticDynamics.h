/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/sync/ExtractSystem.h"
#include "saiga/engine/sync/RenderSink.h"

#include <vector>

namespace Saiga
{
namespace Game
{
/**
 * Render-only dynamic instances that belong to an entity but are not entities (S19b, transitional): the racing wheels,
 * the platformer's limbs until S34. Reconciled in beforeRender over a view: begin(), set() for every entity drawn this
 * frame (its parts are created on first use and recreated when the mesh list changes), end() frees the parts of the
 * entities that were not set (dead, despawned on a client, gone after a load or a Resync). The poses are the frame's
 * (from ExtractSystem::interpolatedPose), sent with prev = curr. clear() in shutdown.
 */
class CosmeticDynamics
{
   public:
    void init(Engine::RenderSink* sink) { renderSink = sink; }
    void begin();
    void set(Engine::Entity e, const std::vector<Engine::MeshId>& meshes, const std::vector<Engine::WorldPose>& poses);
    void end();
    void clear();
    size_t entityCount() const { return entries.size(); }

   private:
    struct Entry
    {
        Engine::Entity entity;
        std::vector<Engine::MeshId> meshes;
        std::vector<Engine::InstanceHandle> handles;
        bool seen = false;
    };
    void free(Entry& e);
    Engine::RenderSink* renderSink = nullptr;
    std::vector<Entry> entries;
};

}  // namespace Game
}  // namespace Saiga
