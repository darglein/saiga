#include "rendering/lighting/box_light.h"


void BoxLightShader::checkUniforms(){
    DirectionalLightShader::checkUniforms();
}





//==================================


BoxLight::BoxLight()
{



}

void BoxLight::createShadowMap(int resX, int resY){
    Light::createShadowMap(resX,resY);
    range = scale.x;
    cam.setProj(-range,range,-range,range,0.01f,scale.z*2.0f);

}


void BoxLight::setDirection(const vec3 &dir){
    direction = glm::normalize(dir);


}

void BoxLight::setFocus(const vec3 &pos){

//    cout<<"todo: BoxLight::setFocus"<<endl;
    cam.setView(pos-direction*range, pos, glm::vec3(0,1,0));
}

void BoxLight::setAmbientIntensity(float ai)
{
    ambientIntensity = ai;
}

void BoxLight::bindUniforms(BoxLightShader &shader, Camera *cam){
    shader.uploadColor(color);
    shader.uploadAmbientIntensity(ambientIntensity);
    shader.uploadModel(model);

//    vec3 viewd = -glm::normalize(vec3((*view)*vec4(direction,0)));
    vec3 viewd = -glm::normalize(vec3((*view)*model[2]));
    shader.uploadDirection(viewd);

    mat4 ip = glm::inverse(cam->proj);
    shader.uploadInvProj(ip);

    if(this->hasShadows()){
        const glm::mat4 biasMatrix(
                    0.5, 0.0, 0.0, 0.0,
                    0.0, 0.5, 0.0, 0.0,
                    0.0, 0.0, 0.5, 0.0,
                    0.5, 0.5, 0.5, 1.0
                    );

        mat4 shadow = biasMatrix*this->cam.proj * this->cam.view * cam->model;
        shader.uploadDepthBiasMV(shadow);

        shader.uploadDepthTexture(shadowmap.depthTexture);
    }

}

void BoxLight::calculateCamera(){
    float length = scale.z;

    vec3 dir = glm::normalize(vec3(this->getDirection()));
    vec3 pos = this->position - dir * length * 0.5f;
    vec3 up = vec3(getUpVector());

//    cout<<this->position<<" "<<dir<<" "<<pos<<endl;
    this->cam.setView(pos,pos+dir,up);

}

bool BoxLight::cullLight(Camera *cam)
{
    //do an exact frustum-frustum intersection if this light casts shadows, else do only a quick check.
//    if(this->hasShadows())
//        this->culled = !this->cam.intersectSAT(cam);
//    else
//        this->culled = cam->sphereInFrustum(this->cam.boundingSphere)==Camera::OUTSIDE;

    culled  = false;
    return culled;
}

