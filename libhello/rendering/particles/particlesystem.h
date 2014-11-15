#pragma once

#include "glm/gtc/random.hpp"

#include "libhello/rendering/particles/particle.h"
#include "libhello/rendering/particles/particle_shader.h"

#include "libhello/opengl/vertexBuffer.h"

#include "libhello/rendering/object3d.h"


class ParticleSystem : public Object3D
{
public:
    ParticleShader* particleShader;
    VertexBuffer<Particle> particleBuffer;
    unsigned int particle_count;
public:
    ParticleSystem(unsigned int particle_count);

    virtual void init();
    void render();
    virtual void bindUniforms() = 0;


};




