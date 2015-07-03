#pragma once

#include "libhello/opengl/basic_shaders.h"

class SAIGA_GLOBAL ParticleShader : public MVPTextureShader {
public:
     GLuint location_timer, location_timestep, location_interpolation;

    ParticleShader(const std::string &multi_file) : MVPTextureShader(multi_file){}

    virtual void checkUniforms();

    void uploadTiming(int tick, float interpolation);
    void uploadTimestep(float timestep);
};


class SAIGA_GLOBAL DeferredParticleShader : public ParticleShader {
public:
     GLuint location_texture_depth;
     GLuint location_cameraParameters;

    DeferredParticleShader(const std::string &multi_file) : ParticleShader(multi_file){}

    virtual void checkUniforms();

    void uploadDepthTexture(raw_Texture *texture);
    void uploadCameraParameters(vec2 cp);
};
