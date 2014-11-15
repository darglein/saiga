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

DirectionalLight::DirectionalLight()
{
}


void DirectionalLight::setDirection(const vec3 &dir){
    direction = glm::normalize(dir);
}

void DirectionalLight::bindUniforms(DirectionalLightShader &shader){
    shader.uploadColor(color);

    vec3 viewd = -glm::normalize(vec3((*view)*vec4(direction,0)));
    shader.uploadDirection(viewd);
}


//void DirectionalLight::drawNoShaderBind(){
//    bindUniforms();
//    buffer.bindAndDraw();
//}



//void DirectionalLight::drawRaw(){
//    buffer.bindAndDraw();
//}
