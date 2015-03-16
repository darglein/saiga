#include "rendering/particles/particleEmitter.h"




ParticleEmitter::ParticleEmitter(){

}

void ParticleEmitter::setParticlesPerTick(float c)
{
    particlesPerTick = c;
}

void ParticleEmitter::setParticlesPerSecond(float c)
{
    particlesPerTick = c * ParticleSystem::secondsPerTick;
}


void ParticleEmitter::apply(ParticleSystem& ps)
{
    time += particlesPerTick;

    int c = 0;
    while(time>=1.0f){
        time -= 1.0f;
        c++;
    }

    if(c>0){
        spawnParticles(c,ps);
        ps.flush();
    }
}




SphericalParticleEmitter::SphericalParticleEmitter()
{

}

void SphericalParticleEmitter::spawnParticles(int count,ParticleSystem& ps)
{

    for(int i=0;i<count;++i){
        Particle p;
        p.position = vec3(this->getPosition());
        p.velocity = vec4(glm::sphericalRand(1.0f),velocity);
        p.radius = radius;
        p.lifetime = lifetime;
        p.fadetime = 0;
        p.image = rand()%4;
        p.scale = 1.0f;
        ps.addParticle(p);
    }

}


ConaParticleEmitter::ConaParticleEmitter()
{

}

void ConaParticleEmitter::spawnParticles(int count,ParticleSystem& ps)
{

    for(int i=0;i<count;++i){
        Particle p;
        p.position = vec3(this->getPosition());
        p.velocity = vec4(sampleCone(coneDirection,glm::radians(coneAngle)),velocity);
        p.radius = radius;
        p.lifetime = lifetime;
        p.image = 4;
        p.force = vec3(0,-1,0);
        p.orientation = Particle::VELOCITY;
        ps.addParticle(p);
    }

}

