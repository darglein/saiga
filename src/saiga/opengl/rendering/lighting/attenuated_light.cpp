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
void AttenuatedLightShader::checkUniforms()
{
    LightShader::checkUniforms();
    location_attenuation = getUniformLocation("attenuation");
}


void AttenuatedLightShader::uploadA(vec3& attenuation, float cutoffRadius)
{
    Shader::upload(location_attenuation, make_vec4(attenuation, cutoffRadius));
}


void LightDistanceAttenuation::renderImGui()
{
    ImGui::InputFloat3("attenuation", &attenuation[0]);
    float c = Evaluate(radius);
    ImGui::Text("Cutoff Intensity: %f", c);
    ImGui::InputFloat("cutoffRadius", &radius);
}


}  // namespace Saiga
