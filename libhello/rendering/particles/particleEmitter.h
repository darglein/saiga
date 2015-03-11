#pragma once

#include "libhello/rendering/particles/particlesystem.h"
#include "libhello/rendering/object3d.h"

class ParticleEmitter : public Object3D{
protected:
    ParticleSystem& particles;

public:

    static float tickRate;

    float particlesPerTick = 0.5f;

    ParticleEmitter(ParticleSystem& particles);

    void update();
    virtual void spawnParticles(int count) = 0;

    void setParticlesPerTick(float c);
    void setParticlesPerSecond(float c);
private:
    float time = 0.0f;
};


class SphericalParticleEmitter : public ParticleEmitter{
public:

    float velocity = 1.0f;
    vec4 color = vec4(1);
    float radius = 0.3f;
    int lifetime = 150;

    SphericalParticleEmitter(ParticleSystem& particles);



    void spawnParticles(int count) override;

};
