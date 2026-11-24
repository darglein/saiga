/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

// The simulation side of the environment (docs/engine_design.md §7.9, §10.7; S36): part of WorldSetup, constant after
// the world exists. The presentation settings (exposure, ambient, clouds, atmosphere) are the app's
// (Vulkan::EnvironmentSettings, EngineApp::environmentSettings).

#pragma once

#include "saiga/engine/lighting/Celestial.h"

namespace Saiga
{
namespace Engine
{
struct EnvironmentSetup
{
    // WorldTime becomes a global and Weather space state (the world-state entity and every space-state entity get
    // them); false: both are registered as plain components that no entity has (D4). Host and clients must agree
    bool enabled = false;
    CelestialConfig celestial;
    // The cloud layer moves with wind * cloudWindScale (worldTimeSystem, the extract)
    double cloudWindScale = 1;
};

}  // namespace Engine
}  // namespace Saiga
