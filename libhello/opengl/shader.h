#ifndef SHADER_H
#define SHADER_H

#include <GL/glew.h>
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

class ShaderLoader : public Loader<Shader>{
public:
    Shader* loadFromFile(const std::string &name);
    template<typename shader_t> shader_t* load(const std::string &name);
    template<typename shader_t> shader_t* loadFromFile(const std::string &name);
    void reload();
};




template<typename shader_t>
shader_t* ShaderLoader::load(const std::string &name){
    shader_t* object;
    //check if already exists
    for(Shader* &obj : objects){
        if(obj->name == name){
            object = dynamic_cast<shader_t*>(obj);
            if(object != nullptr){
                return object;
            }
        }
    }

    bool erg;


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
