/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/camera/camera.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/tostring.h"
#include "saiga/opengl/rendering/deferredRendering/deferredRendering.h"

namespace Saiga
{

mat4 LightBase::viewToLightTransform(const Camera& camera, const Camera& shadowCamera)
{
    // glm like glsl is column major!
    const mat4 biasMatrix = make_mat4(0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.5, 0.5, 1.0);

    // We could also use inverse(camera.view) but using the model matrix is faster
    return biasMatrix * shadowCamera.proj * shadowCamera.view * camera.model;
}

void LightBase::renderImGui()
{
    ImGui::Checkbox("active", &active);
    ImGui::Checkbox("castShadows", &castShadows);
    ImGui::Checkbox("volumetric", &volumetric);
    ImGui::InputFloat("intensity", &intensity, 0.1, 1);
    ImGui::InputFloat("specular intensity", &intensity_specular, 0.1, 1);
    ImGui::SliderFloat("volumetricDensity", &volumetricDensity, 0.0f, 0.1f);
    ImGui::ColorEdit3("colorDiffuse", &colorDiffuse[0]);
    ImGui::ColorEdit3("colorSpecular", &colorSpecular[0]);
    auto str = to_string(visible) + "/" + to_string(selected) + "/" + to_string(culled);
    ImGui::Text("visible/selected/culled: %s", str.c_str());
}

}  // namespace Saiga
