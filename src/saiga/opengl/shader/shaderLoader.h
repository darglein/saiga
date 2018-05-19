/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/shader/shader.h"
#include "saiga/opengl/shader/shaderPartLoader.h"
#include "saiga/util/singleton.h"
#include "saiga/util/loader.h"
#include "saiga/util/assert.h"

namespace Saiga {

class SAIGA_GLOBAL ShaderLoader : public Loader<std::shared_ptr<Shader>,ShaderPart::ShaderCodeInjections> , public Singleton <ShaderLoader>{
    friend class Singleton <ShaderLoader>;
    std::shared_ptr<Shader> loadFromFile(const std::string &name, const ShaderPart::ShaderCodeInjections &params);
    template<typename shader_t> std::shared_ptr<shader_t> loadFromFile(const std::string &name, const ShaderPart::ShaderCodeInjections& sci);
public:
    virtual ~ShaderLoader(){}
    template<typename shader_t> std::shared_ptr<shader_t> load(const std::string &name, const ShaderPart::ShaderCodeInjections& sci=ShaderPart::ShaderCodeInjections());
    template<typename shader_t> std::shared_ptr<shader_t> getLoaded(const std::string &name, const ShaderPart::ShaderCodeInjections& sci=ShaderPart::ShaderCodeInjections());

    void reload();
    bool reload(std::shared_ptr<Shader> shader, const std::string &name, const ShaderPart::ShaderCodeInjections& sci);
};




template<typename shader_t>
std::shared_ptr<shader_t> ShaderLoader::load(const std::string &name, const ShaderPart::ShaderCodeInjections& sci){
    std::shared_ptr<shader_t> object;

    for(data_t &data : objects){
        if(std::get<0>(data)==name && std::get<1>(data)==sci){
            object = std::dynamic_pointer_cast<shader_t>(std::get<2>(data));
            if(object){
                return object;
            }
        }
    }

    std::string fullName = shaderPathes.getFile(name);

    if(fullName.empty()){
        std::cout<<"Could not find file '"<<name<<"'. Make sure it exists and the search pathes are set."<<std::endl;
        SAIGA_ASSERT(0);
    }

    object = loadFromFile<shader_t>(fullName,sci);
    SAIGA_ASSERT(object);
    objects.emplace_back(name,sci,object);
//    std::cout << objects.size() << std::endl;

    return object;
}

template<typename shader_t>
std::shared_ptr<shader_t> ShaderLoader::getLoaded(const std::string &name, const ShaderPart::ShaderCodeInjections& sci){
    std::shared_ptr<shader_t> object;

    for(data_t &data : objects){
        if(std::get<0>(data)==name && std::get<1>(data)==sci){
            object = std::dynamic_pointer_cast<shader_t>(std::get<2>(data));
            if(object){
                return object;
            }
        }
    }

    SAIGA_ASSERT(false && "Shader was not loaded!");

    return nullptr;
}

template<typename shader_t>
std::shared_ptr<shader_t> ShaderLoader::loadFromFile(const std::string &name, const ShaderPart::ShaderCodeInjections& sci){

    ShaderPartLoader spl(name,sci);
    if(spl.load()){
        return spl.createShader<shader_t>();
    }

    return nullptr;
}

}
