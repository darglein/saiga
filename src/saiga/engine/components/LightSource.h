/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/Types.h"
#include "saiga/engine/components/EngineIds.h"
#include "saiga/engine/ecs/Ids.h"
#include "saiga/engine/ecs/Visit.h"
#include "saiga/vulkan/scene/RenderSceneTypes.h"

#include <cmath>
#include <cstdint>
#include <string>

namespace Saiga
{
namespace Engine
{
using Vulkan::LightKind;

// A local light of an entity (docs/engine_design.md §10.7; S36b): a point light, or a spot light along -z of rotation
// (relative to the entity), at offset from the entity's pose. Entity lights are for a moderate number of gameplay
// lights (lamps, torches, vehicle lights); many static lights of a base are S30's piece lights
// (RenderScene::addLights, one list per chunk, no entities). The extract (ExtractSystem) sends it to the render sink:
// added and removed with the component (or with on), changed when a field changes (a write through the world API, a
// replicated change), moved with the entity's interpolated pose every frame. Flicker is a render effect (presentation,
// §10.7; its phase is the environment's). Registered Replicate::HostReliable (rarely changing; on/off toggles
// reliable, not in the snapshots) and saved: host-authoritative, a client's write is not replicated (a player's own
// light, e.g. a torch, is switched by the game's systems from the owner's input; S32 may revisit this).
struct LightSource
{
    static constexpr const char* name = "LightSource";
    static constexpr ComponentId id   = ComponentId(EngineComponent::LightSource);
    static constexpr uint32_t version = 1;

    LightKind kind = LightKind::Point;
    Vec3 color     = Vec3(1, 0.9, 0.8);  // linear
    double lumen   = 800;                // >= 0
    double range   = 10;                 // m, (0, 1000]
    // Spot: the full intensity inside innerAngleDeg, none beyond outerAngleDeg (0 < inner <= outer < 90)
    double innerAngleDeg = 25, outerAngleDeg = 35;
    Vec3 offset             = Vec3::Zero();      // relative to the entity (its rotation applied)
    Quat rotation           = Quat::Identity();  // the spot's -z relative to the entity
    bool on                 = true;
    bool castShadows        = false;
    double priority         = 1;                        // scales the importance (the light budget and the shadow tiles)
    double flickerAmplitude = 0, flickerFrequency = 0;  // [0, 1], Hz (0: none)

    static constexpr const char* kindNames[] = {"point", "spot"};

    template <class V>
    void visit(V& v)
    {
        v("kind", kind, EnumNames{kindNames});
        v("color", color);
        v("lumen", lumen);
        v("range", range);
        v("innerAngleDeg", innerAngleDeg);
        v("outerAngleDeg", outerAngleDeg);
        v("offset", offset);
        v("rotation", rotation);
        v("on", on);
        v("castShadows", castShadows);
        v("priority", priority);
        v("flickerAmplitude", flickerAmplitude);
        v("flickerFrequency", flickerFrequency);
    }

    // The validity of the fields: an empty string, or the first violation. The world API refuses an invalid light
    // (add_component, set_component, spawn and prefab values: checkLightSourceValues); the extract does not send one
    // (a light written in code: list_lights shows it with the reason)
    std::string invalidReason() const
    {
        auto finite = [](double x) { return std::isfinite(x); };
        if (!(range > 0 && range <= 1000)) return "range must be in (0, 1000] m";
        if (!(lumen >= 0) || !finite(lumen)) return "lumen must be finite and >= 0";
        if (kind == LightKind::Spot && !(innerAngleDeg > 0 && innerAngleDeg <= outerAngleDeg && outerAngleDeg < 90))
            return "spot angles must satisfy 0 < innerAngleDeg <= outerAngleDeg < 90";
        if (!finite(color(0)) || !finite(color(1)) || !finite(color(2)) || color.minCoeff() < 0)
            return "color must be finite and >= 0";
        if (!finite(offset(0)) || !finite(offset(1)) || !finite(offset(2))) return "offset must be finite";
        const Vec4 q = rotation.coeffs();
        if (!finite(q(0)) || !finite(q(1)) || !finite(q(2)) || !finite(q(3)) || !(q.norm() > 1e-6))
            return "rotation must be a finite, non-zero quaternion";
        if (!(priority >= 0) || !finite(priority)) return "priority must be finite and >= 0";
        if (!(flickerAmplitude >= 0 && flickerAmplitude <= 1)) return "flickerAmplitude must be in [0, 1]";
        if (!(flickerFrequency >= 0 && flickerFrequency <= 100)) return "flickerFrequency must be in [0, 100] Hz";
        return "";
    }
};

}  // namespace Engine
}  // namespace Saiga
