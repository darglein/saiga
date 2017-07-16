#pragma once

#include "saiga/opengl/opengl.h"
#include "saiga/util/glm.h"
#include "saiga/opengl/shader/shaderpart.h"
#include "saiga/util/fileChecker.h"

#include <vector>
 //for shared pointer

namespace Saiga {

class Shader;

class SAIGA_GLOBAL ShaderPartLoader{
public:


    typedef std::vector<ShaderCodeInjection> ShaderCodeInjections;

	//true if "#line <linenumber> <filename>" should be added
	//at the beginning and around "#includes"
	//Default: false, because ati and intel drivers do not support this
	bool addLineDirectives = false;
    std::string file;
    ShaderCodeInjections injections;

    std::vector<std::shared_ptr<ShaderPart>> shaders;


    ShaderPartLoader();
    ShaderPartLoader(const std::string &file, const ShaderCodeInjections &injections);
    ~ShaderPartLoader();

    bool load();
    bool loadAndPreproccess(const std::string &file, std::vector<std::string> &ret);

    void addShader(std::vector<std::string> &content, GLenum type);

    //combine all loaded shader parts to a shader. the returned shader is linked and ready to use
    template<typename shader_t> std::shared_ptr<shader_t> createShader();

    //like create shader, but the passed shader is updated instead of creating a new one
    void reloadShader(std::shared_ptr<Shader>  shader);
};


template<typename shader_t>
std::shared_ptr<shader_t> ShaderPartLoader::createShader()
{
    if(shaders.size()==0){
        std::cerr<<file<<" does not contain any shaders."<<endl;
        std::cerr<<"Use for example '##GL_FRAGMENT_SHADER' to mark the beginning of a fragment shader."<<endl;
        std::cerr<<"Also make sure this makro is at a beginning of a new line."<<endl;
        return nullptr;
    }

    auto shader = std::make_shared<shader_t>();
    shader->shaders = shaders;
    shader->createProgram();

#ifndef SAIGA_RELEASE
    std::cout<<"Loaded: "<<file<<" ( ";
    for(auto& sp : shaders){
        //todo use glbinding ostream
        std::cout<<(int)sp->type<<" ";
    }
    std::cout<<") Id="<< shader->program << std::endl;
#endif

    shader->name = file;

    return shader;
}

extern SAIGA_GLOBAL FileChecker shaderPathes;

}
