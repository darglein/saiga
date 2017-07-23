/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/rendering/lighting/attenuated_light.h"
#include "saiga/camera/camera.h"

namespace Saiga {

class SAIGA_GLOBAL PointLightShader : public AttenuatedLightShader{
public:
    GLint location_shadowPlanes;

    virtual void checkUniforms();

    void uploadShadowPlanes(float f, float n);
};




class SAIGA_GLOBAL PointLight : public AttenuatedLight
{
    friend class DeferredLighting;
protected:
    Shadowmap shadowmap;
public:
    float shadowNearPlane = 0.1f;
    PerspectiveCamera cam;


    PointLight();

    virtual ~PointLight(){}
    PointLight& operator=(const PointLight& light);



    virtual void bindUniforms(std::shared_ptr<PointLightShader> shader, Camera *cam);


    float getRadius() const;
    virtual void setRadius(float value);


    void createShadowMap(int resX, int resY);

    void bindFace(int face);
    void calculateCamera(int face);


    bool cullLight(Camera *cam);
};

}
