#pragma once

#include "libhello/rendering/object3d.h"
#include "libhello/rendering/particles/particlesystem.h"



class ParticleEffect : public Object3D{
public:
    float velocity = 1.0f;
    vec4 color = vec4(1);
    float radius = 0.1f;
    float lifetime = 1.0f;
public:

    virtual void apply(ParticleSystem& ps) = 0;
};

class ImpactParticleEffect : public ParticleEffect{
public:
    vec3 coneDirection = vec3(0,1,0);
    float coneAngle = 45.0f; //in degrees

    virtual void apply(ParticleSystem& ps) override;
};
