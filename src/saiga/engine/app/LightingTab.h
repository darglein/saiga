/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/export.h"

namespace Saiga
{
namespace Engine
{
class EngineApp;

/**
 * The debug UI's Lighting tab (F5; docs/engine_design.md §17.1; S36), set by the app with DebugUi::setTabContent:
 * Time and weather (the time of day, day and rate, the weather of the view camera's space; writes through setWorldTime
 * / setWeather between steps, so they replicate and appear in the log; with --coop on the hidden host, read-only on a
 * pure client), Sun and moon (directions, illuminance, the shadow light, the sun override) and Exposure (the EV100 of
 * the frame and of the curve, the compensation, the effective exposure and the pre-exposure). Later parts of S36 add
 * the render sections (toolbar, ambient, sky and clouds, cascades, local lights, GPU and memory).
 */
class SAIGA_ENGINE_API LightingTab
{
   public:
    explicit LightingTab(EngineApp& app) : app(app) {}
    // Inside the debug UI's panel (an ImGui frame)
    void draw();

   private:
    void timeAndWeather();
    void sunAndMoon();
    void exposure();
    EngineApp& app;
};

}  // namespace Engine
}  // namespace Saiga
