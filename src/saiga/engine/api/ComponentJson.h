/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// Component values as JSON in the visit form (docs/engine_design.md §14, §16.2; S10, shared with prefab stamping from
// S23a): the checks of an initial or added component, the add with values and the merge of values into a component a
// class added itself. The world API's spawn / add_component and the chunk stamp system use the same path.

#pragma once

#include "saiga/core/util/json11.hpp"
#include "saiga/engine/components/Parent.h"
#include "saiga/engine/components/RenderInstance.h"
#include "saiga/engine/ecs/Entity.h"
#include "saiga/engine/ecs/Ids.h"
#include "saiga/export.h"

#include <string>

namespace Saiga
{
namespace Engine
{
class World;
struct LightSource;

// LightSource values merged into current: an error with the reason when the result is not a valid light
// (LightSource::invalidReason; S36b round 1, finding 7: an invalid light was accepted and silently not rendered)
SAIGA_ENGINE_API bool checkLightSourceValues(const LightSource& current, const json11::Json& values,
                                             std::string& error);

// Components that the generic add / remove / spawn paths do not handle (empty: none); adding: an add (else a remove)
SAIGA_ENGINE_API std::string specialComponent(ComponentId id, bool adding);
// A Parent value from JSON, checked: an alive parent, no cycle, no world-state child (child may be invalid: a new
// entity of spawn)
SAIGA_ENGINE_API bool parentFromJson(const World& w, Entity child, const json11::Json& values, Parent& out,
                                     std::string& error);
// RenderInstance values: the mesh must be registered (aliases resolved); nothing is changed
SAIGA_ENGINE_API bool checkRenderMeshValues(const World& w, const RenderInstance& current, const json11::Json& values,
                                            std::string& error);
// Checks the values of an initial or added component (nothing is changed): special components, RenderInstance meshes,
// LightSource validity, Parent, the fields and their types
SAIGA_ENGINE_API bool checkComponentValues(const World& w, Entity child, ComponentId id, const json11::Json& values,
                                           std::string& error);
// Adds a component with values (checked before; a Parent is checked again: hooks may have changed the world since).
// Between steps and while the world applies commands (World::addDefaultById)
SAIGA_ENGINE_API bool addComponentValues(World& w, Entity e, ComponentId id, const json11::Json& values,
                                         std::string& error);
// The values applied after configure() / onSpawn(): merged into a component the object added itself (mergeJson; a
// Parent is replaced), otherwise added with its default and merged. False with the error if it cannot be applied (the
// entity keeps what was applied before)
SAIGA_ENGINE_API bool mergeComponentValues(World& w, Entity e, ComponentId id, const json11::Json& values,
                                           std::string& error);

}  // namespace Engine
}  // namespace Saiga
