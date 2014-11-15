#pragma once

#include "libhello/opengl/shader.h"

class ParticleShader : public MVPShader {
public:
    ParticleShader(const string &multi_file) : MVPShader(multi_file){}

};
