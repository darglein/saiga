#pragma once

#include "libhello/opengl/basic_shaders.h"

class ParticleShader : public MVPTextureShader {
public:
     GLuint location_timer, location_timestep, location_interpolation;

    ParticleShader(const string &multi_file) : MVPTextureShader(multi_file){}

    virtual void checkUniforms();

    void uploadTiming(int tick, float interpolation);
    void uploadTimestep(float timestep);
};


class DeferredParticleShader : public ParticleShader {
public:
     GLuint location_texture_depth;
     GLuint location_cameraParameters;

    DeferredParticleShader(const string &multi_file) : ParticleShader(multi_file){}

    virtual void checkUniforms();

    void uploadDepthTexture(raw_Texture *texture);
    void uploadCameraParameters(vec2 cp);
};
