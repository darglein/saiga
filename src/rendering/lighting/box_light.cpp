/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/rendering/lighting/box_light.h"

namespace Saiga {

void BoxLightShader::checkUniforms(){
    DirectionalLightShader::checkUniforms();
}





//==================================


BoxLight::BoxLight()
{



}

void BoxLight::createShadowMap(int resX, int resY){
    //    Light::createShadowMap(resX,resY);
    shadowmap.createFlat(resX,resY);
}




void BoxLight::bindUniforms(std::shared_ptr<BoxLightShader> shader, Camera *cam){
    shader->uploadColorDiffuse(colorDiffuse);
    shader->uploadColorSpecular(colorSpecular);
    shader->uploadAmbientIntensity(ambientIntensity);
    shader->uploadModel(model);

    //    vec3 viewd = -glm::normalize(vec3((*view)*vec4(direction,0)));
    vec3 viewd = -glm::normalize(vec3(cam->view*model[2]));
    shader->uploadDirection(viewd);

    mat4 ip = glm::inverse(cam->proj);
    shader->uploadInvProj(ip);

    if(this->hasShadows()){
        shader->uploadDepthBiasMV(viewToLightTransform(*cam,this->cam));
        shader->uploadDepthTexture(shadowmap.getDepthTexture(0));
        shader->uploadShadowMapSize(shadowmap.getSize());
    }

}

void BoxLight::calculateCamera(){
    vec3 dir = glm::normalize(vec3(this->getDirection()));
    vec3 pos = getPosition() ;
    vec3 up = vec3(getUpVector());

    //the camera is centred at the centre of the shadow volume.
    //we define the box only by the sides of the orthographic projection
    cam.setView(pos,pos+dir,up);
    cam.setProj(-scale.x,scale.x,-scale.y,scale.y,-scale.z,scale.z);
}

bool BoxLight::cullLight(Camera *cam)
{
    //do an exact frustum-frustum intersection if this light casts shadows, else do only a quick check.
    if(this->hasShadows())
        this->culled = !this->cam.intersectSAT(cam);
    else
        this->culled = cam->sphereInFrustum(this->cam.boundingSphere)==Camera::OUTSIDE;
    return culled;
}

}
