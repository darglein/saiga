#pragma once

#include "libhello/opengl/basic_shaders.h"

class ParticleShader : public MVPShader {
public:
    ParticleShader(const string &multi_file) : MVPShader(multi_file){}

};
