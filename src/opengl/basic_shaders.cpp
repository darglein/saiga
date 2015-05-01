#include "opengl/basic_shaders.h"
#include "libhello/opengl/framebuffer.h"
void MVPColorShader::checkUniforms(){
    MVPShader::checkUniforms();
    location_color = getUniformLocation("color");
}

void MVPColorShader::uploadColor(const vec4 &color){
    upload(location_color,color);
}

void MVPTextureShader::checkUniforms(){
    MVPShader::checkUniforms();
    location_texture = Shader::getUniformLocation("image");
}


void MVPTextureShader::uploadTexture(raw_Texture *texture){
    texture->bind(0);
    Shader::upload(location_texture,0);
}



void FBShader::checkUniforms(){
    MVPShader::checkUniforms();
    location_texture = getUniformLocation("text");
}

void FBShader::uploadFramebuffer(Framebuffer* fb){
    fb->colorBuffers[0]->bind(0);
    upload(location_texture,0);
}

void DeferredShader::checkUniforms(){
    MVPShader::checkUniforms();
    location_screen_size = getUniformLocation("screen_size");

    location_texture_diffuse = getUniformLocation("deferred_diffuse");
    location_texture_normal = getUniformLocation("deferred_normal");
    location_texture_depth = getUniformLocation("deferred_depth");
    location_texture_position = getUniformLocation("deferred_position");
    location_texture_data = getUniformLocation("deferred_data");

    cout<<"depth position "<<location_texture_depth<<endl;
    cout<<"normal position "<<location_texture_normal<<endl;
}

void DeferredShader::uploadFramebuffer(Framebuffer* fb){

    fb->colorBuffers[0]->bind(0);
    upload(location_texture_diffuse,0);

    fb->colorBuffers[1]->bind(1);
    upload(location_texture_normal,1);

    fb->colorBuffers[2]->bind(2);
    upload(location_texture_data,2);

//    fb->colorBuffers[3]->bind(3);
//    upload(location_texture_position,3);

    fb->depthBuffer->bind(4);
    upload(location_texture_depth,4);
}

void MVPShader::checkUniforms(){
    Shader::checkUniforms();
    location_model = getUniformLocation("model");
    location_view = getUniformLocation("view");
    location_proj = getUniformLocation("proj");
    location_mv = getUniformLocation("MV");
    location_mvp = getUniformLocation("MVP");
}

void MVPShader::uploadAll(const mat4& m1,const mat4& m2,const mat4& m3){
    uploadModel(m1);
    uploadView(m2);
    uploadProj(m3);

    uploadMV(m2*m1);
    uploadMVP(m3*m2*m1);
}

void MaterialShader::checkUniforms(){
    MVPShader::checkUniforms();
    location_colors = getUniformLocation("colors");
    location_textures = getUniformLocation("textures");
    location_use_textures = getUniformLocation("use_textures");
    textures[0] = 0;
    textures[1] = 1;
    textures[2] = 2;
    textures[3] = 3;
    textures[4] = 4;
}

void MaterialShader::uploadMaterial(const Material &material){
    colors[0] = material.Ka;
    colors[1] = material.Kd;
    colors[2] = material.Ks;
    upload(location_colors,3,colors);
    for(int i=0;i<5;i++)
        use_textures[i] = 0.0f;

    if(material.map_Ka){
        material.map_Ka->bind(0);
        use_textures[0] = 1.0f;
    }
    if(material.map_Kd){
        material.map_Kd->bind(1);
        use_textures[1] = 1.0f;
    }
    if(material.map_Ks){
        material.map_Ks->bind(2);
        use_textures[2] = 1.0f;
    }
    if(material.map_d){
        material.map_d->bind(3);
        use_textures[3] = 1.0f;
    }
    if(material.map_bump){
        material.map_bump->bind(4);
        use_textures[4] = 1.0f;
    }
    glUniform1iv(location_textures,5, textures);
    glUniform1fv(location_use_textures,5, use_textures);
}

void TextShader::checkUniforms(){
    MVPShader::checkUniforms();
    location_model = getUniformLocation("model");
    location_proj = getUniformLocation("proj");
    location_color = getUniformLocation("color");
    location_texture = getUniformLocation("text");
}



void TextShader::upload(Texture* texture, const vec3 &color){
    Shader::upload(location_color,color);
    texture->bind(0);
    Shader::upload(location_texture,0);

}
