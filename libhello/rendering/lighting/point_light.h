#pragma once

#include "libhello/opengl/mesh_object.h"
#include "libhello/opengl/shader.h"
#include "libhello/geometry/sphere.h"
#include "libhello/geometry/plane.h"
#include "libhello/geometry/triangle_mesh_generator.h"

#include "libhello/rendering/lighting/light.h"

class PointLightShader : public LightShader{
public:
    GLuint location_position, location_attenuation;
    PointLightShader(const string &multi_file) : LightShader(multi_file){}
    virtual void checkUniforms();
    virtual void upload(const vec3 &pos, float r);
    virtual void upload(vec3 &attenuation);
};



class PointLight : public Light// public LightMesh<PointLight,PointLightShader>
{
public:


    vec3 attenuation;
    float radius;
    Sphere sphere;
    PointLight();
    PointLight(const Sphere &sphere);
    virtual ~PointLight(){}
    PointLight& operator=(const PointLight& light);
    void setAttenuation(float c, float l , float q);
    void setSimpleAttenuation(float d, float cutoff=(1./256.));
    void calculateRadius(float cutoff=(1./256.));

    virtual void bindUniforms(PointLightShader& shader);
    virtual void bindUniformsStencil(MVPShader& shader);


    float getRadius() const;
    virtual void setRadius(float value);

    vec3 getAttenuation() const;
    void setAttenuation(const vec3 &value);


//    void drawNoShaderBind();
//    void drawNoShaderBindStencil();
//    void drawRaw();


};
