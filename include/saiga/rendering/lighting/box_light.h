/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/rendering/lighting/directional_light.h"

namespace Saiga {

class SAIGA_GLOBAL BoxLightShader : public LightShader{
public:

    virtual void checkUniforms();

};

class SAIGA_GLOBAL BoxLight :  public Light
{
    friend class DeferredLighting;
protected:

    Shadowmap shadowmap;

public:
    OrthographicCamera shadowCamera;
    BoxLight();
    virtual ~BoxLight(){}

    void bindUniforms(std::shared_ptr<BoxLightShader> shader, Camera* cam);

    void setView(vec3 pos, vec3 target, vec3 up);

    void createShadowMap(int resX, int resY);

    void calculateCamera();
    bool cullLight(Camera *shadowCamera);

};

}
