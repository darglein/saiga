#pragma once

#include "saiga/opengl/shader/basic_shaders.h"

class SAIGA_GLOBAL ParticleShader : public MVPTextureShader {
public:
    GLint location_timer, location_timestep, location_interpolation;


    virtual void checkUniforms();

    void uploadTiming(int tick, float interpolation);
    void uploadTimestep(float timestep);
};


class SAIGA_GLOBAL DeferredParticleShader : public ParticleShader {
public:
    GLint location_texture_depth;
    GLint location_cameraParameters;


    virtual void checkUniforms();

    void uploadDepthTexture(raw_Texture *texture);
    void uploadCameraParameters(vec2 cp);
};
