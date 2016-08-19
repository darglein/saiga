#pragma once


#include "saiga/rendering/lighting/light.h"
#include "saiga/camera/camera.h"

class SAIGA_GLOBAL DirectionalLightShader : public LightShader{
public:
    GLint location_direction, location_ambientIntensity;
    GLint location_ssaoTexture;

    virtual void checkUniforms();
    void uploadDirection(vec3 &direction);
    void uploadAmbientIntensity(float i);
    void uploadSsaoTexture(raw_Texture* texture);

};

class SAIGA_GLOBAL DirectionalLight :  public Light
{
protected:


    vec3 direction = vec3(0,-1,0);
    float range = 20.0f;
    float ambientIntensity = 0.2f;

public:
    OrthographicCamera cam;
    //    static void createMesh();
    DirectionalLight();
    virtual ~DirectionalLight(){}

    void bindUniforms(DirectionalLightShader& shader, Camera* cam);

    virtual void createShadowMap(int resX, int resY) override;
    void setDirection(const vec3 &dir);
    void setFocus(const vec3 &pos);
    void setAmbientIntensity(float ai);
    float getAmbientIntensity(){return ambientIntensity;}


};


