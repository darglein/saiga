/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ProceduralSkyboxBase.h"

#include "saiga/core/imgui/imgui.h"

#include "internal/noGraphicsAPI.h"
namespace Saiga
{
void ProceduralSkyboxBase::imgui()
{
    ImGui::InputFloat("horizonHeight", &horizonHeight);
    ImGui::InputFloat("distance", &distance);
    ImGui::SliderFloat("sunIntensity", &sunIntensity, 0, 200);
    ImGui::SliderFloat("sunSize", &sunSize, 0, 2);
    ImGui::Direction("sunDir", sunDir);
    ImGui::ColorPicker3("sunColor", &sunColor(0));
    ImGui::ColorPicker3("highSkyColor", &highSkyColor(0));
    ImGui::ColorPicker3("lowSkyColor", &lowSkyColor(0));
}

}  // namespace Saiga
