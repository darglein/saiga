#pragma once


#include "saiga/rendering/lighting/directional_light.h"

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

    virtual void createShadowMap(int resX, int resY) override;

    void calculateCamera();
    bool cullLight(Camera *cam);

};


