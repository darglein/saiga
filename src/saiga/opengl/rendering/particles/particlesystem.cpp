/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/rendering/particles/particlesystem.h"

#include "saiga/core/camera/camera.h"
#include "saiga/core/math/random.h"
#include "saiga/opengl/rendering/particles/particle_shader.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/texture/ArrayTexture2D.h"

namespace Saiga
{
float ParticleSystem::ticksPerSecond = 60.0f;
float ParticleSystem::secondsPerTick = 1.0f / 60.0f;

ParticleSystem::ParticleSystem(unsigned int particleCount) : particleCount(particleCount)
{
    particles.resize(particleCount);
}

void ParticleSystem::init()
{
    for (unsigned int i = 0; i < particleCount; ++i)
    {
        Particle p;
        //        p.position = make_vec4(sphericalRand(15.0f), 1);
        //        p.velocity = make_vec4(sphericalRand(1.0f), 1);
        addParticle(p);
    }

    particleBuffer.set(particles, GL_DYNAMIC_DRAW);
    particleBuffer.setDrawMode(GL_POINTS);

    initialized = true;

    particleShader         = shaderLoader.load<ParticleShader>("geometry/particles.glsl");
    deferredParticleShader = shaderLoader.load<DeferredParticleShader>("geometry/deferred_particles.glsl");
}

void ParticleSystem::reset()
{
    if (!initialized) return;

    for (unsigned int i = 0; i < particleCount; ++i)
    {
        Particle& p = particles[i];
        p           = Particle();
    }

    particleBuffer.updateBuffer(&particles[0], particles.size(), 0);
}

void ParticleSystem::nextTick()
{
    ++tick;
}

void ParticleSystem::update()
{
    if (uploadDataNextUpdate)
    {
        updateParticleBuffer();
        uploadDataNextUpdate = false;
    }
}

void ParticleSystem::interpolate(float interpolation)
{
    this->interpolation = interpolation;
}


void ParticleSystem::render(Camera* cam)
{
    //    std::cout<<tick<<" ParticleSystem::renderr()"<<endl;

    if (blending)
    {
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }

    if(particleShader->bind())
    {
        particleShader->uploadModel(model);
        particleShader->uploadTexture(arrayTexture.get());

        particleShader->uploadTiming(tick, interpolation);
        particleShader->uploadTimestep(secondsPerTick);

        //    particleBuffer.bindAndDraw();
        // draw old particles first, so new ones are on top
        particleBuffer.bind();
        particleBuffer.draw(nextParticle, particleCount - nextParticle);
        particleBuffer.draw(0, nextParticle);
        particleBuffer.unbind();

        particleShader->unbind();
    }

    if (blending)
    {
        glDisable(GL_BLEND);
    }
}

void ParticleSystem::renderDeferred(Camera* cam, std::shared_ptr<TextureBase> detphTexture)
{
    if(deferredParticleShader->bind())
    {
        deferredParticleShader->uploadModel(model);
        deferredParticleShader->uploadTexture(arrayTexture.get());
        deferredParticleShader->uploadDepthTexture(detphTexture);
        deferredParticleShader->uploadTiming(tick, interpolation);
        deferredParticleShader->uploadTimestep(secondsPerTick);

        deferredParticleShader->uploadCameraParameters(vec2(cam->zNear, cam->zFar));
        //    particleBuffer.bindAndDraw();

        particleBuffer.bind();
        particleBuffer.draw(nextParticle, particleCount - nextParticle);
        particleBuffer.draw(0, nextParticle);
        particleBuffer.unbind();

        deferredParticleShader->unbind();
    }
}


void ParticleSystem::addParticle(Particle& p)
{
    p.start = tick + 1;


    particles[nextParticle] = p;
    nextParticle            = (nextParticle + 1) % particleCount;

    newParticles++;
}

// Particle &ParticleSystem::getNextParticle()
//{
//    Particle &p = particles[nextParticle];
//    p = Particle();
//    nextParticle = (nextParticle+1)%particleCount;
//    p.start = tick+1;

//    newParticles++;
//    return p;
//}

void ParticleSystem::updateParticleBuffer()
{
    //    std::cout<<tick<<" ParticleSystem::updateParticleBuffer()"<<endl;

    if (newParticles > particleCount)
    {
        //        std::cout<<"warning: new particles spawned = "<<newParticles<<" , particle system size =
        //        "<<particleCount<<endl;
        int size   = particleCount;
        int offset = 0;
        particleBuffer.updateBuffer(&particles[offset], size, offset);
    }
    else if (nextParticle > saveParticle)
    {
        int size   = (nextParticle - saveParticle);
        int offset = saveParticle;

        particleBuffer.updateBuffer(&particles[offset], size, offset);
    }
    else if (nextParticle < saveParticle)
    {
        int size   = (particleCount - saveParticle);
        int offset = saveParticle;
        particleBuffer.updateBuffer(&particles[offset], size, offset);


        size   = (nextParticle);
        offset = 0;
        particleBuffer.updateBuffer(&particles[0], size, offset);
    }

    saveParticle = nextParticle;
    newParticles = 0;
}

void ParticleSystem::flush()
{
    uploadDataNextUpdate = true;
}

}  // namespace Saiga
