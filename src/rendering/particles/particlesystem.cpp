#include "rendering/particles/particlesystem.h"


ParticleSystem::ParticleSystem(unsigned int particle_count):particle_count(particle_count)
{

}

void ParticleSystem::init(){
    std::vector<Particle> particles(particle_count);
    for(Particle& p : particles){
        p.position = glm::sphericalRand(3.0f);
        p.color = vec4(glm::linearRand(vec3(0),vec3(1)),1);
    }

    this->translateGlobal(vec3(0,8,0));
    createGlBuffer();
}

void ParticleSystem::createGlBuffer(){
    particle_count = particles.size();
    particleBuffer.set(particles);
    particleBuffer.setDrawMode(GL_POINTS);
}

void ParticleSystem::render(){

//    glEnable(GL_BLEND);
//    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);


    glEnable(GL_DEPTH_TEST);
    glDepthMask(GL_TRUE);
    particleShader->bind();
    bindUniforms();
    glPointSize(5.0f);
    particleBuffer.bindAndDraw();
    particleShader->unbind();
    glDepthMask(GL_TRUE);

    glDisable(GL_BLEND);
}


