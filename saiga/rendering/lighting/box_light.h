#pragma once

#include "saiga/rendering/lighting/directional_light.h"

namespace Saiga {

class SAIGA_GLOBAL BoxLightShader : public DirectionalLightShader{
public:

    virtual void checkUniforms();

};

class SAIGA_GLOBAL BoxLight :  public DirectionalLight
{
protected:


public:
    BoxLight();
    virtual ~BoxLight(){}

    void bindUniforms(std::shared_ptr<BoxLightShader> shader, Camera* cam);

    void createShadowMap(int resX, int resY);

    void calculateCamera();
    bool cullLight(Camera *cam);

};

}
