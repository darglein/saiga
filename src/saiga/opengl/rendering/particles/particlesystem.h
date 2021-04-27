/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/geometry/object3d.h"
#include "saiga/opengl/rendering/particles/particle.h"
#include "saiga/opengl/vertexBuffer.h"

#include <memory>
namespace Saiga
{
class TextureBase;
class Camera;
class ParticleShader;
class DeferredParticleShader;
class ArrayTexture2D;

class SAIGA_OPENGL_API ParticleSystem : public Object3D
{
   public:
    std::shared_ptr<ArrayTexture2D> arrayTexture;
    bool blending = true;

    static float secondsPerTick;
    static float ticksPerSecond;

   private:
    std::shared_ptr<ParticleShader> particleShader;
    std::shared_ptr<DeferredParticleShader> deferredParticleShader;

    VertexBuffer<Particle> particleBuffer;
    std::vector<Particle> particles;

    bool initialized = false;

    unsigned int newParticles = 0;
    unsigned int particleCount;
    unsigned int nextParticle = 0;
    unsigned int saveParticle = 0;
    bool uploadDataNextUpdate = false;
    int tick                  = 0;


    float interpolation = 0.0f;



   public:
    ParticleSystem(unsigned int particleCount = 0);

    virtual void init();

    void reset();

    /**
     * @brief nextTick
     * Call this at the start of the update tick so particles added within this tick have the correct start tick
     */
    void nextTick();

    /**
     * @brief update
     * Call this after all particles are added in a tick, to upload the data to the GPU
     */
    void update();
    void interpolate(float interpolation);
    void render(Camera* cam);
    void renderDeferred(Camera* cam, std::shared_ptr<TextureBase> detphTexture);


    void addParticle(Particle& p);

    //    //the returned particle is already added!
    //    Particle& getNextParticle();

    void updateParticleBuffer();
    void flush();
};

}  // namespace Saiga
