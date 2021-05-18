/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/rendering/lighting/point_light.h"

namespace Saiga
{
class SAIGA_OPENGL_API SpotLight : public LightBase, public LightDistanceAttenuation
{
   public:
    struct ShaderData
    {
        vec4 position;       // xyz, w angle
        vec4 colorDiffuse;   // rgb intensity
        vec4 colorSpecular;  // rgb specular intensity
        vec4 attenuation;    // xyz radius
        vec4 direction;      // xyzw
    };

    inline ShaderData GetShaderData()
    {
        ShaderData data;
        float cosa         = cos(radians(angle * 0.95f));  // make border smoother
        data.position      = make_vec4(position, cosa);
        data.colorDiffuse  = make_vec4(colorDiffuse, intensity);
        data.colorSpecular = make_vec4(colorSpecular, intensity_specular);
        data.attenuation   = make_vec4(attenuation, radius);
        data.direction     = make_vec4(direction, 0);
        return data;
    }

    inline ShadowData GetShadowData(Camera* view_point)
    {
        ShadowData sd;
        sd.view_to_light = viewToLightTransform(*view_point, shadowCamera);
        sd.shadow_planes = {shadowCamera.zFar,shadowCamera.zNear};
        sd.inv_shadow_map_size = vec2(1.f / 512, 1.f / 512);
        return sd;
    }

    float shadowNearPlane = 0.01f;
    PerspectiveCamera shadowCamera;
    vec3 direction = vec3(0, -1, 0);
    vec3 position  = vec3(0, 0, 0);
    vec3 getPosition() { return position; }
    void setPosition(const vec3& p) { position = p; }

    float angle = 60.0f;

    /**
     * The default direction of the mesh is negative y
     */

    SpotLight();
    virtual ~SpotLight() {}





    mat4 ModelMatrix();

    void setAngle(float value);
    float getAngle() const { return angle; }

    void setDirection(vec3 dir);

    void calculateCamera();

    bool cullLight(Camera* cam);
    void renderImGui();
};

}  // namespace Saiga
