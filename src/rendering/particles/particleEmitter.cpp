#include "rendering/particles/particleEmitter.h"




ParticleEmitter::ParticleEmitter(ParticleSystem &particles) : particles(particles)
{

}

void ParticleEmitter::setParticlesPerTick(float c)
{
    particlesPerTick = c;
}

void ParticleEmitter::setParticlesPerSecond(float c)
{
    particlesPerTick = c * ParticleSystem::secondsPerTick;
}

void ParticleEmitter::setLifetimeTicks(float c){
    lifetime = c;
}

void ParticleEmitter::setLifetimeSeconds(float c){
    lifetime = c * ParticleSystem::ticksPerSecond;
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


ConaParticleEmitter::ConaParticleEmitter(ParticleSystem &particles) : ParticleEmitter(particles)
{

}

void ConaParticleEmitter::spawnParticles(int count)
{

    for(int i=0;i<count;++i){
        Particle p;
        p.position = vec3(this->getPosition());
        p.velocity = sampleCone(coneDirection,glm::radians(coneAngle));
        p.color = color;
        p.radius = radius;
        p.lifetime = lifetime;
        particles.addParticle(p);
    }

}

