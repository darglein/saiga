#include "rendering/lighting/directional_light.h"


void DirectionalLightShader::checkUniforms(){
    LightShader::checkUniforms();
    location_direction = getUniformLocation("direction");
    location_color = getUniformLocation("color");

}



void DirectionalLightShader::uploadDirection(vec3 &direction){
    Shader::upload(location_direction,direction);
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
    float range = 400.0f;
    cam.setProj(-range,range,-range,range,10.f,800.0f);

}


void DirectionalLight::setDirection(const vec3 &dir){
    direction = glm::normalize(dir);


}

void DirectionalLight::setFocus(const vec3 &pos){


    cam.setView(pos-direction*400.0f, pos, glm::vec3(0,1,0));
}

void DirectionalLight::bindUniforms(DirectionalLightShader &shader, Camera *cam){
    shader.uploadColor(color);

    vec3 viewd = -glm::normalize(vec3((*view)*vec4(direction,0)));
    shader.uploadDirection(viewd);

    const glm::mat4 biasMatrix(
    0.5, 0.0, 0.0, 0.0,
    0.0, 0.5, 0.0, 0.0,
    0.0, 0.0, 0.5, 0.0,
    0.5, 0.5, 0.5, 1.0
    );

    mat4 shadow = biasMatrix*this->cam.proj * this->cam.view * cam->model;
    shader.uploadDepthBiasMV(shadow);

    shader.uploadDepthTexture(shadowmap.depthBuffer.depthBuffer);
}


