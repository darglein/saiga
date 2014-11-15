#ifndef SHADER_H
#define SHADER_H

#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <SDL2/SDL_opengl.h>
#include <GL/glu.h>
#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <fstream>

#include "libhello/util/glm.h"
#include "libhello/opengl/vertex.h"
#include "libhello/rendering/material.h"
#include "libhello/util/loader.h"
#include "libhello/opengl/framebuffer.h"

using std::cerr;


class Shader{
public:
    string name;
    string shaderPath;
    string typeToName(int type);

    Shader();
    virtual ~Shader();
    Shader(const string &multi_file);

    //Shader loading utility programs
    void printProgramLog( GLuint program );
    void printShaderLog( GLuint shader );

    bool addMultiShaderFromFile(const string &multi_file);
    GLuint addShader(const char* content, int type);
    GLuint addShaderFromFile(const char* file, int type);
    GLuint createProgram();
    GLint getUniformLocation(const char* name);



public:

    GLuint program, vertShader,geoShader,fragShader;
    static Shader* getShader(int id); //time o(1)
    static void reloadAll();
    static void clearShaders(); //deleted all shaders
    static void createShaders(); //creates all shaders


    bool reload();
    void bind();
    void unbind();


    void upload(int location, const mat4 &m);
    void upload(int location, const vec4 &v);
    void upload(int location, const vec3 &v);
    void upload(int location, const vec2 &v);
    void upload(int location, const int &v);
    void upload(int location, const float &f);
    //array uploads
    void upload(int location, int count, mat4* m);
    void upload(int location, int count, vec4* v);
    void upload(int location, int count, vec3* v);
    void upload(int location, int count, vec2* v);

    virtual void checkUniforms(){}
};

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

class ShaderLoader : public Loader<Shader>{
public:
    Shader* loadFromFile(const std::string &name);
    template<typename shader_t> shader_t* load(const std::string &name);
    template<typename shader_t> shader_t* loadFromFile(const std::string &name);
    void reload();
};




template<typename shader_t>
shader_t* ShaderLoader::load(const std::string &name){
    //check if already exists
    for(Shader* &object : objects){
        if(object->name == name)
            return dynamic_cast<shader_t*>(object);
    }

    bool erg;
    shader_t* object;

    for(std::string &path : locations){
        std::string complete_path = path + "/" + name;
        object = loadFromFile<shader_t>(complete_path);
        if (object){
            object->name = name;
            std::cout<<"Loaded from file: "<<complete_path<<std::endl;
            objects.push_back(object);
            return object;
        }
    }

    std::cout<<"Failed to load "<<name<<"!!!"<<std::endl;
    return NULL;
}

template<typename shader_t>
shader_t* ShaderLoader::loadFromFile(const std::string &name){
    shader_t* shader = new shader_t(name);
    if(shader->reload()){
        return shader;
    }
    delete shader;
    return NULL;
}



#endif // SHADER_H
