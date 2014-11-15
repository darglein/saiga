#pragma once

#include "libhello/opengl/mesh_object.h"

#include "libhello/geometry/sphere.h"
#include "libhello/geometry/plane.h"
#include "libhello/geometry/triangle_mesh_generator.h"

#include "libhello/rendering/lighting/light.h"

class DirectionalLightShader : public LightShader{
public:
    GLuint location_direction;
    DirectionalLightShader(const string &multi_file) : LightShader(multi_file){}
    virtual void checkUniforms();
    void uploadDirection(vec3 &direction);
};

class DirectionalLight :  public Light //public LightMesh<DirectionalLight,DirectionalLightShader>
{
protected:


    vec3 direction;
public:
    const mat4 *view;
//    static void createMesh();
    DirectionalLight();
    virtual ~DirectionalLight(){}
    void bindUniforms(DirectionalLightShader& shader);
    void setDirection(const vec3 &dir);

//    void drawNoShaderBind();
//    void drawRaw();
};


