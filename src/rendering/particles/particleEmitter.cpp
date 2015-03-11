#include "rendering/particles/particleEmitter.h"

float ParticleEmitter::tickRate = 1.0f/60.0f;

ParticleEmitter::ParticleEmitter(ParticleSystem &particles) : particles(particles)
{

}

void ParticleEmitter::setParticlesPerTick(float c)
{
    particlesPerTick = c;
}

void ParticleEmitter::setParticlesPerSecond(float c)
{
    particlesPerTick = c * ParticleEmitter::tickRate;
}

void ParticleEmitter::update()
{
    time += particlesPerTick;

    int c = 0;
    while(time>=1.0f){
        time -= 1.0f;
        c++;
    }

    if(c>0){
        spawnParticles(c);
        particles.flush();
    }
}




SphericalParticleEmitter::SphericalParticleEmitter(ParticleSystem &particles) : ParticleEmitter(particles)
{

}

void SphericalParticleEmitter::spawnParticles(int count)
{

    for(int i=0;i<count;++i){
        Particle p;
        p.position = vec3(this->getPosition());
        p.velocity = glm::sphericalRand(velocity);
        p.color = color;
        p.radius = radius;
        p.lifetime = lifetime;
        particles.addParticle(p);
    }

}


