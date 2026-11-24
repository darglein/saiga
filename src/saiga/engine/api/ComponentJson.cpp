/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/engine/api/ComponentJson.h"

#include "saiga/engine/components/LightSource.h"
#include "saiga/engine/components/Persistent.h"
#include "saiga/engine/components/RenderInstance.h"
#include "saiga/engine/components/Transform.h"
#include "saiga/engine/ecs/VisitJson.h"
#include "saiga/engine/gameplay/Behaviour.h"
#include "saiga/engine/net/NetOwner.h"
#include "saiga/engine/world/World.h"

namespace Saiga
{
namespace Engine
{
using Json = json11::Json;

namespace
{
const Json& emptyObject()
{
    static const Json empty = Json::object{};
    return empty;
}

bool stateEntity(const Registry& reg, Entity e)
{
    return reg.flags(e) & (Registry::kWorldState | Registry::kSpaceState);
}

// p is e or one of e's descendants (a Parent link from e to p would close a cycle)
bool isSelfOrDescendant(const World& w, Entity e, Entity p)
{
    const uint32_t limit = w.registry().slotCount();
    for (uint32_t n = 0; n <= limit && p.valid(); ++n)
    {
        if (p == e) return true;
        const Parent* link = w.tryRead<Parent>(p);
        p                  = link ? link->entity : Entity{};
    }
    return false;
}

bool uniformScale(const Vec3& s)
{
    return s(1) == s(0) && s(2) == s(0);
}

}  // namespace

// RenderInstance values: the mesh must be registered (aliases resolved); nothing is changed (S19b)
bool checkRenderMeshValues(const World& w, const RenderInstance& current, const Json& values, std::string& error)
{
    RenderInstance r = current;
    if (!fromJson(values, r, &error, true)) return false;
    Vulkan::MeshId id = 0;
    if (!w.resolveRenderMesh(r.mesh, id))
    {
        error = "RenderInstance: the mesh \"" + r.mesh + "\" is not registered";
        return false;
    }
    return true;
}

bool checkLightSourceValues(const LightSource& current, const Json& values, std::string& error)
{
    LightSource l = current;
    if (!fromJson(values, l, &error, true)) return false;
    const std::string invalid = l.invalidReason();
    if (invalid.empty()) return true;
    error = "LightSource: " + invalid;
    return false;
}

// A Parent value from JSON, checked: an alive parent, no cycle, no world-state child (child may be invalid: a new
// entity of spawn)
bool parentFromJson(const World& w, Entity child, const Json& values, Parent& out, std::string& error)
{
    Parent p = out;
    if (!fromJson(values, p, &error, true)) return false;
    if (!w.alive(p.entity))
    {
        error = "Parent: the parent entity is not alive";
        return false;
    }
    if (stateEntity(w.registry(), p.entity))
    {
        error = "Parent: a world-state entity is not a parent";
        return false;
    }
    if (child.valid() && stateEntity(w.registry(), child))
    {
        error = "Parent: world-state entities cannot have a parent";
        return false;
    }
    if (child.valid() && isSelfOrDescendant(w, child, p.entity))
    {
        error = "Parent: the link would form a cycle";
        return false;
    }
    out = p;
    return true;
}

// Components that the generic add/remove/spawn paths do not handle
std::string specialComponent(ComponentId id, bool adding)
{
    if (id == Transform::id) return "Transform: every entity has one (use set_transform; its space: move_to_space)";
    if (id == Behaviour::id && adding) return "Behaviour: added by spawn with a class";
    if (id == Persistent::id && adding) return "Persistent: assigned by makePersistent";
    if (id == NetOwner::id) return "NetOwner: set by setOwner on the host before the spawn (read-only)";
    return {};
}

// Checks the values of an initial or added component (nothing is changed)
bool checkComponentValues(const World& w, Entity child, ComponentId id, const Json& values, std::string& error)
{
    const std::string special = specialComponent(id, true);
    if (!special.empty())
    {
        error = special;
        return false;
    }
    if (id == RenderInstance::id && !values.is_null() && !checkRenderMeshValues(w, RenderInstance(), values, error))
        return false;
    if (id == LightSource::id && !values.is_null() && !checkLightSourceValues(LightSource(), values, error))
        return false;
    if (id == Parent::id)
    {
        Parent p;
        return parentFromJson(w, child, values.is_null() ? emptyObject() : values, p, error);
    }
    if (values.is_null()) return true;
    return w.registry().poolById(id)->checkJson(values, error);
}

// Adds a component with values (checked before; a Parent is checked again: hooks may have changed the world since)
bool addComponentValues(World& w, Entity e, ComponentId id, const Json& values, std::string& error)
{
    if (id == Parent::id)
    {
        Parent p;
        if (!parentFromJson(w, e, values.is_null() ? emptyObject() : values, p, error)) return false;
        w.add<Parent>(e, p);
        return true;
    }
    w.addDefaultById(e, id);
    bool ok = true;
    if (!values.is_null()) ok = w.registry().poolById(id)->mergeJson(e.index, values, error);
    // The add path of the name (S19b)
    if (id == RenderInstance::id) w.renderMeshWritten(e, true);
    return ok;
}

bool mergeComponentValues(World& w, Entity e, ComponentId id, const Json& values, std::string& error)
{
    if (w.registry().entityMask(e).test(id))
    {
        if (id == Parent::id)
        {
            Parent p = w.read<Parent>(e);
            if (!parentFromJson(w, e, values, p, error)) return false;
            w.remove<Parent>(e);
            w.add<Parent>(e, p);
            return true;
        }
        if (values.is_null()) return true;
        // LightSource: the values merged into the class's light must give a valid light (spawn checked them against
        // the default light only; CodeRabbit #86)
        if (id == LightSource::id && !checkLightSourceValues(w.read<LightSource>(e), values, error)) return false;
        // RenderInstance: the name write goes through renderMeshWritten (its cached ID and name record; the values
        // were checked before)
        std::string previousMesh;
        if (id == RenderInstance::id) previousMesh = w.read<RenderInstance>(e).mesh;
        const bool ok = w.registry().poolById(id)->mergeJson(e.index, values, error);
        if (id == RenderInstance::id) w.renderMeshWritten(e, false, &previousMesh);
        return ok;
    }
    if (id == RenderInstance::id && !uniformScale(w.read<Transform>(e).scale()))
    {
        // The class set a non-uniform scale: not rendered
        error = "an entity with RenderInstance has a uniform scale (x = y = z); the class set a non-uniform one";
        return false;
    }
    return addComponentValues(w, e, id, values, error);
}

}  // namespace Engine
}  // namespace Saiga
