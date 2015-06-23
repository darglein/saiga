#pragma once

#include "libhello/opengl/shader.h"
#include "libhello/util/singleton.h"
#include "libhello/util/loader.h"

class ShaderLoader : public Loader<Shader,Shader::ShaderCodeInjections> , public Singleton <ShaderLoader>{
    friend class Singleton <ShaderLoader>;
public:
    virtual ~ShaderLoader(){}
    Shader* loadFromFile(const std::string &name);
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

//    for(Shader* &obj : objects){
//        if(obj->name == name){
//            object = dynamic_cast<shader_t*>(obj);
//            if(object != nullptr){
//                return object;
//            }
//        }
//    }


    for(std::string &path : locations){
        std::string complete_path = path + "/" + name;
        object = loadFromFile<shader_t>(complete_path,path,sci);
        if (object){
            object->name = name;
            std::cout<<"Loaded from file: "<<complete_path<<std::endl;
//            objects.push_back(object);
            objects.emplace_back(name,sci,object);
            return object;
        }
    }

    std::cout<<"Failed to load "<<name<<"!!!"<<std::endl;
    exit(0);
    return NULL;
}

template<typename shader_t>
shader_t* ShaderLoader::loadFromFile(const std::string &name, const std::string &prefix, const Shader::ShaderCodeInjections& sci){
    shader_t* shader = new shader_t(name);
    shader->prefix = prefix;
    shader->injections = sci;
    if(shader->reload()){
        return shader;
    }
    delete shader;
    return NULL;
}
