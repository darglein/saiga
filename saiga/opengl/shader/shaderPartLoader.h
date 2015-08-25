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



    ShaderPartLoader();
    ShaderPartLoader(const std::string &file, const std::string &prefix, const ShaderCodeInjections &injections);
    virtual ~ShaderPartLoader();




};




