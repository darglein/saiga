#pragma once

#include "saiga/opengl/shader/shader.h"

#include "saiga/opengl/vertex.h"
#include "saiga/rendering/material.h"

class Framebuffer;

class SAIGA_GLOBAL MVPShader : public Shader{
public:
    MVPShader(const std::string &multi_file) : Shader(multi_file){}
    int location_model, location_view, location_proj;
    int location_mvp, location_mv;
    virtual void checkUniforms();

    void uploadAll(const mat4& m1,const mat4& m2,const mat4& m3);
    void uploadMVP(const mat4& matrix){upload(location_mvp,matrix);}
    void uploadMV(const mat4& matrix){upload(location_mv,matrix);}
    void uploadModel(const mat4& matrix){upload(location_model,matrix);}
    void uploadView(const mat4& matrix){upload(location_view,matrix);}
    void uploadProj(const mat4& matrix){upload(location_proj,matrix);}
};

class SAIGA_GLOBAL MVPColorShader : public MVPShader{
public:
    int location_color;
    MVPColorShader(const std::string &multi_file) : MVPShader(multi_file){}
    virtual void checkUniforms();
    virtual void uploadColor(const vec4 &color);
};

class SAIGA_GLOBAL MVPTextureShader : public MVPShader{
public:
    int location_texture;
    MVPTextureShader(const std::string &multi_file) : MVPShader(multi_file){}
    virtual void checkUniforms();
    virtual void uploadTexture(raw_Texture* texture);
};


class SAIGA_GLOBAL FBShader : public MVPShader{
public:
    int location_texture;
    FBShader(const std::string &multi_file) : MVPShader(multi_file){}
    virtual void checkUniforms();
    virtual void uploadFramebuffer(Framebuffer* fb);
};

class SAIGA_GLOBAL DeferredShader : public FBShader{
public:
    int location_screen_size;
    int location_texture_diffuse,location_texture_normal,location_texture_depth,location_texture_data;
    DeferredShader(const std::string &multi_file) : FBShader(multi_file){}
    virtual void checkUniforms();
    void uploadFramebuffer(Framebuffer* fb);
    void uploadScreenSize(vec2 sc){Shader::upload(location_screen_size,sc);}
};



class SAIGA_GLOBAL MaterialShader : public MVPShader{
public:
    int location_colors;
    int location_textures, location_use_textures;
    vec3 colors[3]; //ambiend, diffuse, specular
    GLint textures[5]; //ambiend, diffuse, specular, alpha, bump
    float use_textures[5]; //1.0 if related texture is valid
    MaterialShader(const std::string &multi_file) : MVPShader(multi_file){}
    virtual void checkUniforms();
    void uploadMaterial(const Material &material);

};

class SAIGA_GLOBAL TextShader : public MVPShader {
public:
    int location_color, location_texture,location_strokeColor;
    TextShader(const std::string &multi_file) : MVPShader(multi_file){}
    virtual void checkUniforms();

    void upload(Texture* texture, const vec4 &color,const vec4 &strokeColor);
};

class SAIGA_GLOBAL TextShaderFade : public TextShader {
public:
    int location_fade;
    TextShaderFade(const std::string &multi_file) : TextShader(multi_file){}
    virtual void checkUniforms();

    void uploadFade(float fade);
};



