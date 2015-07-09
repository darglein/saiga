#include "libhello/opengl/shaderLoader.h"


Shader* ShaderLoader::loadFromFile(const std::string &name, const Shader::ShaderCodeInjections &params){
    Shader* shader = new Shader(name);
    if(shader->reload()){
        return shader;
    }
    delete shader;
    return nullptr;
}

void ShaderLoader::reload(){
    for(auto &object : objects){
        std::get<2>(object)->reload();
    }
}
