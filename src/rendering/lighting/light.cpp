#include "rendering/lighting/light.h"

void LightShader::checkUniforms(){
    DeferredShader::checkUniforms();
    location_color = getUniformLocation("color");
}


void LightShader::uploadColor(vec4 &color){
    Shader::upload(location_color,color);
}

void LightShader::uploadColor(vec3 &color, float intensity){
    vec4 c = vec4(color,intensity);
    Shader::upload(location_color,c);
}
