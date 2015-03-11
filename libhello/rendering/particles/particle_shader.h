#pragma once

#include "libhello/opengl/basic_shaders.h"

class ParticleShader : public MVPShader {
public:
     GLuint location_timer, location_timestep, location_interpolation;

    ParticleShader(const string &multi_file) : MVPShader(multi_file){}

    virtual void checkUniforms();

    void uploadTiming(int tick, float interpolation);
    void uploadTimestep(float timestep);
};
