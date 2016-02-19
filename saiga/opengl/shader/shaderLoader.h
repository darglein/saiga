#pragma once

#include "saiga/opengl/shader/shader.h"
#include "saiga/opengl/shader/shaderPartLoader.h"
#include "saiga/util/singleton.h"
#include "saiga/util/loader.h"


class SAIGA_GLOBAL ShaderLoader : public Loader<Shader,ShaderPart::ShaderCodeInjections> , public Singleton <ShaderLoader>{
    friend class Singleton <ShaderLoader>;
public:
    virtual ~ShaderLoader(){}
    Shader* loadFromFile(const std::string &name, const ShaderPart::ShaderCodeInjections &params);
    template<typename shader_t> shader_t* load(const std::string &name, const ShaderPart::ShaderCodeInjections& sci=ShaderPart::ShaderCodeInjections());
    template<typename shader_t> shader_t* loadFromFile(const std::string &name, const std::string &prefix, const ShaderPart::ShaderCodeInjections& sci);

    void reload();
    bool reload(Shader* shader, const std::string &name, const std::string &prefix, const ShaderPart::ShaderCodeInjections& sci);
};




template<typename shader_t>
shader_t* ShaderLoader::load(const std::string &name, const ShaderPart::ShaderCodeInjections& sci){



    shader_t* object;

    for(data_t &data : objects){
        if(std::get<0>(data)==name && std::get<1>(data)==sci){
            object = dynamic_cast<shader_t*>(std::get<2>(data));
            if(object != nullptr){
                return object;
            }
        }
    }

    for(std::string &prefix : locations){
        object = loadFromFile<shader_t>(name,prefix,sci);
        if (object){
            //            std::cout<<"Loaded from file: "<<prefix + "/" + name<<std::endl;
            objects.emplace_back(name,sci,object);
            return object;
        }
    }

    std::cout<<"Failed to load "<<name<<"!!!"<<std::endl;
    assert(0);
    return nullptr;
}

template<typename shader_t>
shader_t* ShaderLoader::loadFromFile(const std::string &name, const std::string &prefix, const ShaderPart::ShaderCodeInjections& sci){

    ShaderPartLoader spl(name,prefix,sci);
    if(spl.load()){
        return spl.createShader<shader_t>();
    }

    return nullptr;
}
