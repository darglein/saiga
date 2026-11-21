/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/engine/components/EngineIds.h"
#include "saiga/engine/ecs/Ids.h"

#include <cstdint>
#include <string>

namespace Saiga
{
namespace Engine
{
// What an entity renders (docs/engine_design.md §10.1, §10.3; S19b): the name of a mesh registered in the World
// (World::registerMesh, by name like DataRef, so visit needs no World context and saves keep names). An empty name:
// tracked, not drawn (the extract keeps the entity's pose pair for interpolatedPose, e.g. a character before S34).
// Registered Replicate::HostReliable (rarely changing; names stay out of the snapshots) and saved. Every write of the
// name goes through World::renderMeshWritten (World::setRenderMesh, the world API, the add path); get, tryGet and
// mutable views are refused at compile time.
struct RenderInstance
{
    static constexpr const char* name = "RenderInstance";
    static constexpr ComponentId id   = ComponentId(EngineComponent::RenderInstance);
    static constexpr uint32_t version = 1;

    std::string mesh;
    // The MeshId of mesh (0: none or not resolved), set by World::renderMeshWritten and loads; not visited (the name is
    // the state)
    uint16_t cachedMesh = 0;

    RenderInstance() = default;
    explicit RenderInstance(std::string meshName) : mesh(std::move(meshName)) {}

    template <class V>
    void visit(V& v)
    {
        v("mesh", mesh);
    }
};

}  // namespace Engine
}  // namespace Saiga
