#pragma once

#include "saiga/opengl/shader/shader.h"
#include "saiga/opengl/shader/shaderPartLoader.h"
#include "saiga/util/singleton.h"
#include "saiga/util/loader.h"


class SAIGA_GLOBAL ShaderLoader : public Loader<Shader,Shader::ShaderCodeInjections> , public Singleton <ShaderLoader>{
    friend class Singleton <ShaderLoader>;
public:
    virtual ~ShaderLoader(){}
    Shader* loadFromFile(const std::string &name, const Shader::ShaderCodeInjections &params);
    template<typename shader_t> shader_t* load(const std::string &name, const Shader::ShaderCodeInjections& sci=Shader::ShaderCodeInjections());
    template<typename shader_t> shader_t* loadFromFile(const std::string &name, const std::string &prefix, const Shader::ShaderCodeInjections& sci);
    void reload();
};




template<typename shader_t>
shader_t* ShaderLoader::load(const std::string &name, const Shader::ShaderCodeInjections& sci){



    shader_t* object;

    for(data_t &data : objects){
        if(std::get<0>(data)==name && std::get<1>(data)==sci){
            object = dynamic_cast<shader_t*>(std::get<2>(data));
            if(object != nullptr){
                return object;
            }
        }
    }

    for(std::string &path : locations){
        std::string complete_path = path + "/" + name;
        object = loadFromFile<shader_t>(complete_path,path,sci);
        if (object){
            object->name = name;
            std::cout<<"Loaded from file: "<<complete_path<<std::endl;
            objects.emplace_back(name,sci,object);
            return object;
        }
    }

    std::cout<<"Failed to load "<<name<<"!!!"<<std::endl;
    exit(0);
    return nullptr;
}

template<typename shader_t>
shader_t* ShaderLoader::loadFromFile(const std::string &name, const std::string &prefix, const Shader::ShaderCodeInjections& sci){
    shader_t* shader = new shader_t(name);
    shader->prefix = prefix;
    shader->injections = sci;
    if(shader->reload()){

        //TODO:
        ShaderPartLoader(name,prefix,sci);

        return shader;
    }
    delete shader;
    return nullptr;
}
