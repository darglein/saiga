#pragma once

#include "saiga/opengl/basic_shaders.h"
#include "saiga/rendering/lighting/shadowmap.h"
#include "saiga/rendering/object3d.h"

class SAIGA_GLOBAL LightShader : public DeferredShader{
public:
    int location_color; //rgba, rgb=color, a=intensity [0,1]
    int location_depthBiasMV, location_depthTex,location_readShadowMap;
    int location_invProj;
    LightShader(const std::string &multi_file) : DeferredShader(multi_file){}
    virtual void checkUniforms();
    void uploadColor(vec4 &color);
    void uploadColor(vec3 &color, float intensity);
    void uploadDepthBiasMV(mat4 &mat);
    void uploadDepthTexture(raw_Texture* texture);
    void uploadShadow(float shadow);
    void uploadInvProj(mat4 &mat);
};


class SAIGA_GLOBAL Light  : public Object3D
{
protected:
    bool visible=true, active=true, selected=false, culled=false;

    //shadow map
    bool castShadows=false;


public:
//    raw_Texture* dummyTexture = nullptr; //0x0 texture to fix an ati error
    Shadowmap shadowmap;
    vec4 color;


    Light(){}
    Light(const vec3 &color, float intensity){setColor(color);setIntensity(intensity);}
    Light(const vec4 &color){setColor(color);}
    virtual ~Light(){}

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
    void enableShadows() {if(shadowmap.isInitialized()) castShadows=true;}
    void disableShadows() {castShadows=false;}

    virtual void createShadowMap(int resX, int resY);
    void bindShadowMap();
    void unbindShadowMap();

    bool shouldCalculateShadowMap(){return castShadows&&active&&!culled;}
    bool shouldRender(){return active&&!culled;}

    void bindUniformsStencil(MVPShader &shader);
};


