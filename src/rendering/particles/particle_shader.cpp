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
