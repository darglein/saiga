#include "rendering/particles/particle_shader.h"



void ParticleShader::checkUniforms()
{
    MVPTextureShader::checkUniforms();

    location_timer = getUniformLocation("timer");

    location_timestep =  getUniformLocation("timestep");
    location_interpolation = getUniformLocation("interpolation");
}

void ParticleShader::uploadTiming(int tick, float interpolation)
{
    Shader::upload(location_timer,tick);
    Shader::upload(location_interpolation,interpolation);
}

void ParticleShader::uploadTimestep(float timestep)
{
    Shader::upload(location_timestep,timestep);
}


//===============================================================

void DeferredParticleShader::checkUniforms()
{
    ParticleShader::checkUniforms();

    location_texture_depth = getUniformLocation("depthTexture");
}



void DeferredParticleShader::uploadDepthTexture(raw_Texture *texture){
    texture->bind(1);
    Shader::upload(location_texture_depth,1);
}
