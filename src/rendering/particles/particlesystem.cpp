#include "rendering/particles/particlesystem.h"


ParticleSystem::ParticleSystem(unsigned int particle_count):particle_count(particle_count)
{

}

void ParticleSystem::init(){
    particles.resize(particle_count);
    for(Particle& p : particles){
        p.position = glm::sphericalRand(15.0f);
        p.color = vec4(glm::linearRand(vec3(0),vec3(1)),1);
        p.radius = 0.2f;

    }

    this->translateGlobal(vec3(0,8,0));
    createGlBuffer();
}

void ParticleSystem::createGlBuffer(){
    particle_count = particles.size();
    particleBuffer.set(particles);
    particleBuffer.setDrawMode(GL_POINTS);
}

void ParticleSystem::render(Camera *cam){


    particleShader->bind();

    particleShader->uploadAll(model,cam->view,cam->proj);

    particleBuffer.bindAndDraw();
    particleShader->unbind();

}

void ParticleSystem::bindUniforms()
{

}


