/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/util/assert.h"
#include "point_light.h"
#include "internal/noGraphicsAPI.h"
namespace Saiga
{
PointLight::PointLight()
{
    polygon_offset = vec2(2.0, 100.0);
}


mat4 PointLight::ModelMatrix()
{
    vec3 scale    = make_vec3(radius);
    vec3 position = (this->position);

    return createTRSmatrix(position, quat::Identity(), scale);
}

struct CameraDirection
{
    vec3 Target;
    vec3 Up;
};

static const CameraDirection gCameraDirections[] = {
    { vec3(1.0f, 0.0f, 0.0f), vec3(0.0f, -1.0f, 0.0f)},
    { vec3(-1.0f, 0.0f, 0.0f), vec3(0.0f, -1.0f, 0.0f)},
    { vec3(0.0f, 1.0f, 0.0f), vec3(0.0f, 0.0f, 1.0f)},
    { vec3(0.0f, -1.0f, 0.0f), vec3(0.0f, 0.0f, -1.0f)},
    { vec3(0.0f, 0.0f, 1.0f), vec3(0.0f, -1.0f, 0.0f)},
    { vec3(0.0f, 0.0f, -1.0f), vec3(0.0f, -1.0f, 0.0f)}};

void PointLight::calculateCamera(int face)
{
    vec3 pos(position);
    vec3 dir(gCameraDirections[face].Target);
    vec3 up(gCameraDirections[face].Up);
    shadowCamera.setView(pos, pos + dir, up);
    shadowCamera.setProj(90.0f, 1, shadowNearPlane, radius);
}

bool PointLight::cullLight(Camera* cam)
{
    Sphere s(position, radius);
    this->culled = cam->sphereInFrustum(s) == Camera::OUTSIDE;
    //    this->culled = false;
    //    std::cout<<culled<<endl;
    return culled;
}

void PointLight::renderImGui()
{
    LightBase::renderImGui();
    LightDistanceAttenuation::renderImGui();
    ImGui::InputFloat("shadowNearPlane", &shadowNearPlane);
}

}  // namespace Saiga
