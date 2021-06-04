/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/rendering/lighting/spot_light.h"

#include "saiga/core/imgui/imgui.h"

#include "internal/noGraphicsAPI.h"

namespace Saiga
{
SpotLight::SpotLight()
{
    polygon_offset = vec2(2.0, 100.0);
}


void SpotLight::calculateCamera()
{
    mat4 M             = ModelMatrix();
    shadowCamera.model = M;
    shadowCamera.updateFromModel();
    shadowCamera.setProj(2 * angle, 1, shadowNearPlane, radius);
}

mat4 SpotLight::ModelMatrix()
{
    float l = tan(radians(angle)) * radius;
    vec3 s(l, l, radius);
    quat rot = rotation(vec3(0, 0, -1), normalize(direction));
    return createTRSmatrix((position), rot, s);
}

void SpotLight::setAngle(float value)
{
    this->angle = value;
}

void SpotLight::setDirection(vec3 dir)
{
    direction = dir;
}

bool SpotLight::cullLight(Camera* cam)
{
    // do an exact frustum-frustum intersection if this light casts shadows, else do only a quick check.
    if (this->castShadows)
        this->culled = !this->shadowCamera.intersectSAT(*cam);
    else
        this->culled = cam->sphereInFrustum(this->shadowCamera.boundingSphere) == Camera::OUTSIDE;

    return culled;
}


void SpotLight::renderImGui()
{
    LightBase::renderImGui();
    LightDistanceAttenuation::renderImGui();
    ImGui::SliderFloat("Angle", &angle, 0, 85);

    ImGui::InputFloat("shadowNearPlane", &shadowNearPlane);
}

}  // namespace Saiga
