#pragma once

#include "saiga/rendering/particles/particle.h"
#include "saiga/opengl/vertexBuffer.h"
#include "saiga/rendering/object3d.h"

class raw_Texture;
class Camera;
class ParticleShader;
class DeferredParticleShader;
class ArrayTexture2D;

class SAIGA_GLOBAL ParticleSystem : public Object3D
{
public:
    ArrayTexture2D* arrayTexture;

    ParticleShader* particleShader;
    DeferredParticleShader* deferredParticleShader;

    VertexBuffer<Particle> particleBuffer;
    std::vector<Particle> particles;

    bool initialized = false;

    unsigned int newParticles = 0 ;
    unsigned int particleCount;
    unsigned int nextParticle = 0;
    unsigned int saveParticle = 0;
    bool uploadDataNextUpdate = false;
    int tick = 0;

    bool blending = true;

    float interpolation = 0.0f;

    static float secondsPerTick;
    static float ticksPerSecond;

public:
    ParticleSystem(unsigned int particleCount=0);

    virtual void init();

    void reset();

    void update();
    void interpolate(float interpolation);
    void render(Camera* cam);
    void renderDeferred(Camera* cam, raw_Texture *detphTexture);


    void addParticle(Particle &p);

//    //the returned particle is already added!
//    Particle& getNextParticle();

    void updateParticleBuffer();
    void flush();
};




