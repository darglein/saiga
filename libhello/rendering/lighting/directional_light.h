#pragma once

#include "libhello/opengl/mesh_object.h"

#include "libhello/geometry/sphere.h"
#include "libhello/geometry/plane.h"
#include "libhello/geometry/triangle_mesh_generator.h"

#include "libhello/rendering/lighting/light.h"
#include "libhello/camera/camera.h"
#include "libhello/opengl/framebuffer.h"

class DirectionalLightShader : public LightShader{
public:
    GLuint location_direction;
    GLuint location_depthBiasMV, location_depthTex;

    DirectionalLightShader(const string &multi_file) : LightShader(multi_file){}
    virtual void checkUniforms();
    void uploadDirection(vec3 &direction);
    void uploadDepthBiasMV(mat4 &mat);
    void uploadDepthTexture(raw_Texture* texture);
};

class DirectionalLight :  public Light
{
protected:


    vec3 direction;


public:
    Framebuffer depthBuffer;
     OrthographicCamera cam;
    const mat4 *view;
//    static void createMesh();
    DirectionalLight();
    virtual ~DirectionalLight(){}

    void bindUniforms(DirectionalLightShader& shader, Camera* cam);

    void setDirection(const vec3 &dir);
    void setFocus(const vec3 &pos);


};


