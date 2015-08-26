#pragma once

#include "saiga/rendering/lighting/light.h"
#include "saiga/camera/camera.h"

class SAIGA_GLOBAL PointLightShader : public LightShader{
public:
    GLint location_position, location_attenuation, location_shadowPlanes;

    virtual void checkUniforms();
    virtual void upload(const vec3 &pos, float r);
    virtual void upload(vec3 &attenuation);
    void uploadShadowPlanes(float f, float n);
};



class SAIGA_GLOBAL PointLight : public Light// public LightMesh<PointLight,PointLightShader>
{
public:
    PerspectiveCamera cam;
//    cube_Texture* cubeMap;


    vec3 attenuation;
    float radius;
    Sphere sphere;
    PointLight();
    PointLight(const Sphere &sphere);
    virtual ~PointLight(){}
    PointLight& operator=(const PointLight& light);
    void setAttenuation(float c, float l , float q);
    void setSimpleAttenuation(float d, float cutoff=(1./256.));

    void setLinearAttenuation(float r, float drop);

    void calculateRadius(float cutoff=(1./256.));

    virtual void bindUniforms(PointLightShader& shader, Camera *cam);


    float getRadius() const;
    virtual void setRadius(float value);

    vec3 getAttenuation() const;
    void setAttenuation(const vec3 &value);

    virtual void createShadowMap(int resX, int resY) override;
    void bindFace(int face);
    void calculateCamera(int face);

    float getAttenuation(float r);

    bool cullLight(Camera *cam);
};
