#pragma once

#include "libhello/opengl/shader.h"

#include "libhello/opengl/vertex.h"
#include "libhello/rendering/material.h"

class Framebuffer;

class MVPShader : public Shader{
public:
    MVPShader(const string &multi_file) : Shader(multi_file){}
    GLuint location_model, location_view, location_proj;
    GLuint location_mvp, location_mv;
    virtual void checkUniforms();

    void uploadAll(const mat4& m1,const mat4& m2,const mat4& m3);
    void uploadMVP(const mat4& matrix){upload(location_mvp,matrix);}
    void uploadMV(const mat4& matrix){upload(location_mv,matrix);}
    void uploadModel(const mat4& matrix){upload(location_model,matrix);}
    void uploadView(const mat4& matrix){upload(location_view,matrix);}
    void uploadProj(const mat4& matrix){upload(location_proj,matrix);}
};

class MVPColorShader : public MVPShader{
public:
    GLuint location_color;
    MVPColorShader(const string &multi_file) : MVPShader(multi_file){}
    virtual void checkUniforms();
    virtual void uploadColor(const vec4 &color);
};

class MVPTextureShader : public MVPShader{
public:
    GLuint location_texture;
    MVPTextureShader(const string &multi_file) : MVPShader(multi_file){}
    virtual void checkUniforms();
    virtual void uploadTexture(raw_Texture* texture);
};


class FBShader : public MVPShader{
public:
    GLuint location_texture;
    FBShader(const string &multi_file) : MVPShader(multi_file){}
    virtual void checkUniforms();
    virtual void uploadFramebuffer(Framebuffer* fb);
};

class DeferredShader : public FBShader{
public:
    GLuint location_screen_size;
    GLuint location_texture_diffuse,location_texture_normal,location_texture_position,location_texture_depth;
    DeferredShader(const string &multi_file) : FBShader(multi_file){}
    virtual void checkUniforms();
    void uploadFramebuffer(Framebuffer* fb);
    void uploadScreenSize(vec2 sc){Shader::upload(location_screen_size,sc);}
};



class MaterialShader : public MVPShader{
public:
    GLuint location_colors;
    GLuint location_textures, location_use_textures;
    vec3 colors[3]; //ambiend, diffuse, specular
    GLint textures[5]; //ambiend, diffuse, specular, alpha, bump
    float use_textures[5]; //1.0 if related texture is valid
    MaterialShader(const string &multi_file) : MVPShader(multi_file){}
    virtual void checkUniforms();
    void uploadMaterial(const Material &material);

};

class TextShader : public MVPShader {
public:
    GLuint location_color, location_texture;
    TextShader(const string &multi_file) : MVPShader(multi_file){}
    virtual void checkUniforms();

    void upload(Texture* texture, const vec3 &color);
};


