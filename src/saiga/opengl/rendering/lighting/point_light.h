/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/camera/camera.h"
#include "saiga/opengl/rendering/lighting/attenuated_light.h"

namespace Saiga
{
class SAIGA_OPENGL_API PointLight : public LightBase, public LightDistanceAttenuation
{
   public:
    struct ShaderData
    {
        vec4 position;       // xyz, w unused
        vec4 colorDiffuse;   // rgb intensity
        vec4 colorSpecular;  // rgb specular intensity
        vec4 attenuation;    // xyz radius
    };

    inline ShaderData GetShaderData()
    {
        ShaderData data;
        data.position      = make_vec4(position, 0.0f);
        data.colorDiffuse  = make_vec4(colorDiffuse, intensity);
        data.colorSpecular = make_vec4(colorSpecular, intensity_specular);
        data.attenuation   = make_vec4(attenuation, radius);
        return data;
    }

    inline ShadowData GetShadowData(Camera* view_point)
    {
        ShadowData sd;
        sd.view_to_light = view_point->model;
        sd.shadow_planes = {shadowCamera.zFar,shadowCamera.zNear};
        sd.inv_shadow_map_size = vec2(1.f / 512, 1.f / 512);
        return sd;
    }

    vec3 position;

    void setPosition(const vec3& p) { position = p; }
    vec3 getPosition() { return position; }
    float shadowNearPlane = 0.1f;
    PerspectiveCamera shadowCamera;

    PointLight();
    virtual ~PointLight() {}
    PointLight& operator=(const PointLight& light) = delete;



    mat4 ModelMatrix();



    void bindFace(int face);
    void calculateCamera(int face);


    bool cullLight(Camera* camera);
    void renderImGui();
};

}  // namespace Saiga
