#include "libhello/rendering/lighting/spot_light.h"


void SpotLightShader::checkUniforms(){
    PointLightShader::checkUniforms();
    location_direction = getUniformLocation("direction");
    location_angle = getUniformLocation("angle");
}



void SpotLightShader::uploadDirection(vec3 &direction){
    Shader::upload(location_direction,direction);
}

void SpotLightShader::uploadAngle(float angle){
    Shader::upload(location_angle,angle);
}

SpotLight::SpotLight():PointLight(){

}


void SpotLight::calculateCamera(){
    vec3 dir = vec3(this->getUpVector());
    vec3 pos = vec3(getPosition());
    vec3 up = vec3(getRightVector());
    this->cam.setView(pos,pos-dir,up);
    this->cam.setProj(2*angle,1,1.0,400.0);
}

void SpotLight::bindUniforms(SpotLightShader &shader, Camera *cam){
    shader.uploadColor(color);
    shader.uploadModel(model);
    shader.upload(sphere.pos,sphere.r);
    shader.upload(attenuation);


    const glm::mat4 biasMatrix(
                0.5, 0.0, 0.0, 0.0,
                0.0, 0.5, 0.0, 0.0,
                0.0, 0.0, 0.5, 0.0,
                0.5, 0.5, 0.5, 1.0
                );

    mat4 shadow = biasMatrix*this->cam.proj * this->cam.view * cam->model;
    shader.uploadDepthBiasMV(shadow);

    shader.uploadDepthTexture(shadowmap.depthBuffer.depthBuffer);

    vec3 dir = vec3(this->getUpVector());
    shader.uploadDirection(dir);

    float c = glm::cos(glm::radians(angle*0.95f)); //make border smoother

    shader.uploadAngle(c);

//    const glm::mat4 biasMatrix(
//                0.5, 0.0, 0.0, 0.0,
//                0.0, 0.5, 0.0, 0.0,
//                0.0, 0.0, 0.5, 0.0,
//                0.5, 0.5, 0.5, 1.0
//                );

//    mat4 shadow = biasMatrix*this->cam.proj * this->cam.view * cam->model;
//    shader.uploadDepthBiasMV(shadow);

//    shader.uploadDepthTexture(depthBuffer.depthBuffer);
}

void SpotLight::bindUniformsStencil(MVPShader& shader){

    PointLight::bindUniformsStencil(shader);
}

void SpotLight::recalculateScale(){
    float l = glm::tan(glm::radians(angle))*radius;

    vec3 scale(l,radius,l);

    this->setScale(scale);
}

void SpotLight::setRadius(float value)
{
    radius = value;
    recalculateScale();
}

void SpotLight::setAngle(float value){
    this->angle = value;
    recalculateScale();
}
