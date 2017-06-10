#include "saiga/rendering/lighting/spot_light.h"


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
    cam.setView(pos,pos-dir,up);
    cam.setProj(2*angle,1,shadowNearPlane,radius);

}

void SpotLight::bindUniforms(std::shared_ptr<SpotLightShader> shader, Camera *cam){
    PointLight::bindUniforms(shader,cam);


    vec3 dir = vec3(this->getUpVector());
    shader->uploadDirection(dir);

    float c = glm::cos(glm::radians(angle*0.95f)); //make border smoother

    shader->uploadAngle(c);


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

void SpotLight::createShadowMap(int resX, int resY) {
//    Light::createShadowMap(resX,resY);
//    float farplane = 50.0f;
    shadowmap.createFlat(resX,resY);
}

void SpotLight::setAngle(float value){
    this->angle = value;
    recalculateScale();
}

void SpotLight::setDirection(vec3 dir)
{
    rot = glm::rotation(glm::normalize(dir),vec3(0,-1,0));
}

bool SpotLight::cullLight(Camera *cam)
{
    //do an exact frustum-frustum intersection if this light casts shadows, else do only a quick check.
    if(this->hasShadows())
        this->culled = !this->cam.intersectSAT(cam);
    else
        this->culled = cam->sphereInFrustum(this->cam.boundingSphere)==Camera::OUTSIDE;

    return culled;
}
