#pragma once

#include "libhello/rendering/lighting/point_light.h"

class SpotLightShader : public PointLightShader{
public:
    GLuint location_direction,location_angle;
    SpotLightShader(const std::string &multi_file) : PointLightShader(multi_file){}
    virtual void checkUniforms();
    void uploadDirection(vec3 &direction);
    void uploadAngle(float angle);
};



class SpotLight :  public PointLight
{
private:
    float angle=60.0f;
public:

    SpotLight();
    virtual ~SpotLight(){}
    void bindUniforms(SpotLightShader& shader, Camera *cam);
    void bindUniformsStencil(MVPShader& shader) override;


    void calculateCamera();

    void setRadius(float value) override;

    virtual void createShadowMap(int resX, int resY) override;

    void recalculateScale();
    void setAngle(float value);
    float getAngle() const{return angle;}

    bool cullLight(Camera *cam);
};
