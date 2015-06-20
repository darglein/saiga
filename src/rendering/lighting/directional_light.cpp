#include "rendering/lighting/directional_light.h"


void DirectionalLightShader::checkUniforms(){
    LightShader::checkUniforms();
    location_direction = getUniformLocation("direction");
    location_ambientIntensity = getUniformLocation("ambientIntensity");
    location_ssaoTexture = getUniformLocation("ssaoTex");
}



void DirectionalLightShader::uploadDirection(vec3 &direction){
    Shader::upload(location_direction,direction);
}

void DirectionalLightShader::uploadAmbientIntensity(float i)
{
    Shader::upload(location_ambientIntensity,i);
}

void DirectionalLightShader::uploadSsaoTexture(raw_Texture *texture)
{

        texture->bind(6);
        Shader::upload(location_ssaoTexture,6);
}


//==================================

//void DirectionalLight::createMesh(){
//    Plane p(vec3(0),vec3(0,1,0));
//    auto* m = TriangleMeshGenerator::createFullScreenQuadMesh();
//    m->createBuffers(buffer);
//}

DirectionalLight::DirectionalLight():cam("Sun")
{



}

void DirectionalLight::createShadowMap(int resX, int resY){
    Light::createShadowMap(resX,resY);
    range = 20.0f;
    cam.setProj(-range,range,-range,range,1.f,50.0f);

}


void DirectionalLight::setDirection(const vec3 &dir){
    direction = glm::normalize(dir);


}

void DirectionalLight::setFocus(const vec3 &pos){

//    cout<<"todo: DirectionalLight::setFocus"<<endl;
    cam.setView(pos-direction*range, pos, glm::vec3(0,1,0));
}

void DirectionalLight::setAmbientIntensity(float ai)
{
    ambientIntensity = ai;
}

void DirectionalLight::bindUniforms(DirectionalLightShader &shader, Camera *cam){
    shader.uploadColor(color);
    shader.uploadAmbientIntensity(ambientIntensity);

    vec3 viewd = -glm::normalize(vec3((*view)*vec4(direction,0)));
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


