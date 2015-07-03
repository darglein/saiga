#pragma once


#include "libhello/rendering/lighting/directional_light.h"
#include "libhello/camera/camera.h"

class SAIGA_GLOBAL BoxLightShader : public DirectionalLightShader{
public:

    BoxLightShader(const std::string &multi_file) : DirectionalLightShader(multi_file){}
    virtual void checkUniforms();

};

class SAIGA_GLOBAL BoxLight :  public DirectionalLight
{
protected:


public:
//    static void createMesh();
    BoxLight();
    virtual ~BoxLight(){}

    void bindUniforms(BoxLightShader& shader, Camera* cam);

    virtual void createShadowMap(int resX, int resY) override;
    void setDirection(const vec3 &dir);
    void setFocus(const vec3 &pos);
    void setAmbientIntensity(float ai);

    void calculateCamera();
    bool cullLight(Camera *cam);

};


