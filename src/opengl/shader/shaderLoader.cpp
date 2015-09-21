#include "saiga/opengl/shader/shaderLoader.h"


Shader* ShaderLoader::loadFromFile(const std::string &name, const ShaderPart::ShaderCodeInjections &params){
//    Shader* shader = new Shader(name);
//    if(shader->reload()){
//        return shader;
//    }
//    delete shader;
    return nullptr;
}

void ShaderLoader::reload(){
    cout<<"ShaderLoader::reload"<<endl;
    for(auto &object : objects){
        auto name = std::get<0>(object);
        auto sci = std::get<1>(object);
        auto shader = std::get<2>(object);

        for(std::string &prefix : locations){
            if (reload(shader,name,prefix,sci)){
               break;
            }
        }

    }


}

bool ShaderLoader::reload(Shader *shader, const std::string &name, const std::string &prefix, const ShaderPart::ShaderCodeInjections &sci)
{
    ShaderPartLoader spl(name,prefix,sci);
    if(spl.load()){
       spl.reloadShader(shader);
       return true;
    }
    return false;
}
