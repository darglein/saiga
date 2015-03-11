#pragma once

#include "libhello/rendering/particles/particlesystem.h"
#include "libhello/rendering/object3d.h"

class ParticleEmitter : public Object3D{
protected:
    ParticleSystem& particles;

public:

    float velocity = 1.0f;
    vec4 color = vec4(1);
    float radius = 0.3f;
    float lifetime = 150;




    float particlesPerTick = 0.5f;

    ParticleEmitter(ParticleSystem& particles);

    void update();
    virtual void spawnParticles(int count) = 0;

    void setParticlesPerTick(float c);
    void setParticlesPerSecond(float c);

    void setLifetimeTicks(float c);
    void setLifetimeSeconds(float c);

private:
    float time = 0.0f;
};


class SphericalParticleEmitter : public ParticleEmitter{
public:
    SphericalParticleEmitter(ParticleSystem& particles);
    void spawnParticles(int count) override;
};


class ConaParticleEmitter : public ParticleEmitter{
public:
    vec3 coneDirection = vec3(0,1,0);
    float coneAngle = 45.0f; //in degrees
    ConaParticleEmitter(ParticleSystem& particles);
    void spawnParticles(int count) override;
};
