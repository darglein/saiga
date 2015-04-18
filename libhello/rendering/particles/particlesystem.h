#pragma once



#include "libhello/rendering/particles/particle.h"
#include "libhello/rendering/particles/particle_shader.h"

#include "libhello/opengl/vertexBuffer.h"

#include "libhello/rendering/object3d.h"
#include "libhello/camera/camera.h"
#include "libhello/opengl/texture/arrayTexture.h"

class ParticleSystem : public Object3D
{
public:
    ArrayTexture2D* arrayTexture;

    ParticleShader* particleShader;
    DeferredParticleShader* deferredParticleShader;

    VertexBuffer<Particle> particleBuffer;
    std::vector<Particle> particles;
    bool initialized = false;

    unsigned int particleCount;
    unsigned int nextParticle = 0;
    unsigned int saveParticle = 0;
    bool uploadDataNextUpdate = false;
    int tick = 0;

    bool blending = true;

    static float secondsPerTick;
     static float ticksPerSecond;

public:
    ParticleSystem(unsigned int particleCount=0);

    virtual void init();

    void reset();

    void update();
    void render(Camera* cam, float interpolation);
    void renderDeferred(Camera* cam, float interpolation, raw_Texture *d);


    void addParticle(Particle &p);

    //the returned particle is already added!
    Particle& getNextParticle();

    void updateParticleBuffer();
    void flush();
};




