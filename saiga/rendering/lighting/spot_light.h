#pragma once

#include "saiga/rendering/lighting/point_light.h"

class SAIGA_GLOBAL SpotLightShader : public PointLightShader{
public:
    GLint location_direction,location_angle;

    virtual void checkUniforms();
    void uploadDirection(vec3 &direction);
    void uploadAngle(float angle);
};



class SAIGA_GLOBAL SpotLight :  public PointLight
{
private:
    float angle=60.0f;
public:

    SpotLight();
    virtual ~SpotLight(){}
    void bindUniforms(SpotLightShader& shader, Camera *cam);




    void setRadius(float value) override;

    virtual void createShadowMap(int resX, int resY) override;

    void recalculateScale();
    void setAngle(float value);
    float getAngle() const{return angle;}

    void calculateCamera();
    bool cullLight(Camera *cam);
};
