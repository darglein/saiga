/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/rendering/lighting/attenuated_light.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/assert.h"

namespace Saiga
{
void LightDistanceAttenuation::renderImGui()
{
    ImGui::SliderFloat("atten. a", &attenuation[0], 0, 2);
    ImGui::SliderFloat("atten. b", &attenuation[1], 0, 2);
    ImGui::SliderFloat("atten. c", &attenuation[2], 0, 16);
    ImGui::InputFloat3("attenuation", &attenuation[0]);
    float c = Evaluate(radius);
    ImGui::Text("Cutoff Intensity: %f", c);
    ImGui::InputFloat("cutoffRadius", &radius);
}


}  // namespace Saiga
