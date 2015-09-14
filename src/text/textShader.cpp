#include "saiga/text/textShader.h"

void TextShader::checkUniforms(){
    MVPShader::checkUniforms();
    location_model = getUniformLocation("model");
    location_proj = getUniformLocation("proj");
    location_color = getUniformLocation("color");
    location_strokeColor = getUniformLocation("strokeColor");
    location_texture = getUniformLocation("text");
}



void TextShader::upload(Texture* texture, const vec4 &color, const vec4 &strokeColor){
    Shader::upload(location_color,color);
    Shader::upload(location_strokeColor,strokeColor);
    Shader::upload(location_texture,texture,0);
}


void TextShaderFade::checkUniforms()
{
    TextShader::checkUniforms();
     location_fade = getUniformLocation("fade");
}

void TextShaderFade::uploadFade(float fade)
{
    Shader::upload(location_fade,fade);
}
