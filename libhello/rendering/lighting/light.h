#pragma once

#include "libhello/opengl/mesh_object.h"
#include "libhello/opengl/basic_shaders.h"
#include "libhello/geometry/sphere.h"
#include "libhello/geometry/plane.h"
#include "libhello/geometry/triangle_mesh.h"
#include "libhello/opengl/framebuffer.h"

class LightShader : public DeferredShader{
public:
    GLuint location_color; //rgba, rgb=color, a=intensity [0,1]
    GLuint location_depthBiasMV, location_depthTex;
    LightShader(const string &multi_file) : DeferredShader(multi_file){}
    virtual void checkUniforms();
    void uploadColor(vec4 &color);
    void uploadColor(vec3 &color, float intensity);
    void uploadDepthBiasMV(mat4 &mat);
    void uploadDepthTexture(raw_Texture* texture);
};


class Light  : public Object3D
{
protected:
    bool visible=true, active=true, selected=false;

    //shadow map
    bool castShadows=false;
    int shadowResX,shadowResY;

public:
    Framebuffer depthBuffer;
    vec4 color;


    Light(){}
    Light(const vec3 &color, float intensity){setColor(color);setIntensity(intensity);}
    Light(const vec4 &color){setColor(color);}

    void setColor(const vec3 &color){this->color = vec4(color,this->color.w);}
    void setColor(const vec4 &color){this->color = color;}
    void setIntensity(float f){this->color.w = f;}
    void addIntensity(float f){this->color.w += f;}





    vec3 getColor(){return vec3(color);}
    float getIntensity(){return color.w;}

    void setActive(bool active){this->active=active;}
    bool isActive(){return active;}
    void setVisible(bool visible){this->visible=visible;}
    bool isVisible(){return visible;}
    void setSelected(bool selected){this->selected=selected;}
    bool isSelected(){return selected;}


    bool hasShadows() const {return castShadows;}
    void enableShadows() {castShadows=true;}
    void disableShadows() {castShadows=false;}

    void createShadowMap(int resX, int resY);
    void bindShadowMap();
    void unbindShadowMap();
};


