#pragma once

#include "saiga/opengl/opengl.h"
#include "saiga/util/glm.h"
#include "saiga/opengl/shader/shaderpart.h"

#include <vector>




class SAIGA_GLOBAL ShaderPartLoader{
public:


    typedef std::vector<ShaderCodeInjection> ShaderCodeInjections;

    std::string file;
    std::string prefix;
    ShaderCodeInjections injections;

    std::vector<std::shared_ptr<ShaderPart>> shaders;


    ShaderPartLoader();
    ShaderPartLoader(const std::string &file, const std::string &prefix, const ShaderCodeInjections &injections);
    ~ShaderPartLoader();

    bool load();
    std::vector<std::string> loadAndPreproccess(const std::string &file);
    void addShader(std::vector<std::string> &content, GLenum type);

    //combine all loaded shader parts to a shader. the returned shader is linked and ready to use
    template<typename shader_t> shader_t* createShader();
};


template<typename shader_t>
shader_t* ShaderPartLoader::createShader()
{
    if(shaders.size()==0)
        return nullptr;

    shader_t* shader = new shader_t();
    shader->shaders = shaders;
    shader->createProgram();

    std::cout<<"Loaded: "<<prefix + "/" + file<<" ( ";
    for(auto& sp : shaders){
        std::cout<<sp->type<<" ";
    }
    std::cout<<")"<<std::endl;


    return shader;
}




