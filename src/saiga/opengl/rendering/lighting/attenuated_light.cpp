/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/rendering/lighting/attenuated_light.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/assert.h"
#include "saiga/opengl/error.h"

#include "attenuated_light.h"

namespace Saiga
{
void LightDistanceAttenuation::renderImGui()
{
    ImGui::InputFloat3("attenuation", &attenuation[0]);
    float c = Evaluate(radius);
    ImGui::Text("Cutoff Intensity: %f", c);
    ImGui::InputFloat("cutoffRadius", &radius);
}


}  // namespace Saiga
