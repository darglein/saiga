#pragma once

#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/rendering/lighting/shadowmap.h"
#include "saiga/rendering/object3d.h"

class SAIGA_GLOBAL LightShader : public DeferredShader{
public:
    GLint location_lightColorDiffuse, location_lightColorSpecular; //rgba, rgb=color, a=intensity [0,1]
    GLint location_depthBiasMV, location_depthTex,location_readShadowMap;
    GLint location_shadowMapSize; //vec4(w,h,1/w,1/h)
    GLint location_invProj;

    virtual void checkUniforms();

    void uploadColorDiffuse(vec4 &color);
    void uploadColorDiffuse(vec3 &color, float intensity);

    void uploadColorSpecular(vec4 &color);
    void uploadColorSpecular(vec3 &color, float intensity);

    void uploadDepthBiasMV(mat4 &mat);
    void uploadDepthTexture(raw_Texture* texture);
    void uploadShadow(float shadow);
    void uploadShadowMapSize(float w, float h);
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
    vec4 colorDiffuse = vec4(1);
    vec4 colorSpecular = vec4(1);



    Light(){}
    Light(const vec3 &color, float intensity){setColorDiffuse(color);setIntensity(intensity);}
    Light(const vec4 &color){setColorDiffuse(color);}
    virtual ~Light(){}

    void setColorDiffuse(const vec3 &color){this->colorDiffuse = vec4(color,this->colorDiffuse.w);}
    void setColorDiffuse(const vec4 &color){this->colorDiffuse = color;}
    void setColorSpecular(const vec3 &color){this->colorSpecular = vec4(color,this->colorSpecular.w);}
    void setColorSpecular(const vec4 &color){this->colorSpecular = color;}
    void setIntensity(float f){this->colorDiffuse.w = f;}
    void addIntensity(float f){this->colorDiffuse.w += f;}






    vec3 getColorDiffuse(){return vec3(colorDiffuse);}
    float getIntensity(){return colorDiffuse.w;}

    void setActive(bool _active){this->active= _active;}
    bool isActive(){return active;}
    void setVisible(bool _visible){this->visible= _visible;}
    bool isVisible(){return visible;}
    void setSelected(bool _selected){this->selected= _selected;}
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


