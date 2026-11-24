/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The environment extract (docs/engine_design.md §10.3, §10.7; S36): the render-time EnvironmentState of a space from
// WorldTime, the space's Weather, the celestial model and the app's EnvironmentSettings. Called by the app once per
// rendered frame for the view camera's space; reads WorldTime and Weather, writes nothing.

#pragma once

#include "saiga/core/math/ChunkId.h"
#include "saiga/engine/components/Weather.h"
#include "saiga/engine/components/WorldTime.h"
#include "saiga/engine/lighting/Celestial.h"
#include "saiga/engine/lighting/EnvironmentSetup.h"
#include "saiga/export.h"
#include "saiga/vulkan/scene/LightingTypes.h"
#include "saiga/vulkan/scene/RenderSceneTypes.h"

namespace Saiga
{
namespace Engine
{
class World;

// The extract with the double values it came from (get_lighting, the Lighting tab)
struct EnvironmentSample
{
    bool hasTime      = false;  // the world has WorldTime (EnvironmentSetup::enabled); else the defaults
    bool enclosed     = false;  // the space is enclosed (SpaceTemplate::kind)
    double renderTime = 0;      // World::time() + (alpha - 1) * dt
    WorldTime time;             // at the render time (interpolated on the host, extrapolated on a client), wrapped
    Weather weather;            // normalised; cloudOffset at the render time
    CelestialState celestial;   // at the render time; the sun override replaces the sun's direction
    ShadowLight shadowLight = ShadowLight::None;
    float curveEv100        = 15;  // the exposure curve at the sun elevation (without the compensation)
    Vulkan::EnvironmentState state;
};

// The render time is tr = World::time() + (alpha - 1) * dt (on a client World::time() is the estimate of the host
// time, HostClock); the time of day is timeOfDay + rate * (tr - validAt) / 3600, wrapped, and the cloud offset moves
// with wind * (tr - validAt) * cloudWindScale: on the host this interpolates between the previous and the current
// step, on a client it extrapolates from the snapshot that wrote the values. An enclosed space has no sun, moon, sky or
// sky ambient: its template's ambient and EV100. Without WorldTime / Weather (EnvironmentSetup::enabled false) the
// component defaults (noon, cloudiness 0.3)
SAIGA_ENGINE_API EnvironmentSample sampleEnvironment(const World& world, SpaceId space, double alpha,
                                                     const Vulkan::EnvironmentSettings& settings,
                                                     const EnvironmentSetup& setup);
SAIGA_ENGINE_API Vulkan::EnvironmentState environmentState(const World& world, SpaceId space, double alpha,
                                                           const Vulkan::EnvironmentSettings& settings,
                                                           const EnvironmentSetup& setup);

}  // namespace Engine
}  // namespace Saiga
