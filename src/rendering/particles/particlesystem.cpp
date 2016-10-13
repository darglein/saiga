#include "saiga/rendering/particles/particlesystem.h"
#include "saiga/opengl/texture/arrayTexture.h"
#include "saiga/rendering/particles/particle_shader.h"
#include "saiga/camera/camera.h"
#include "saiga/opengl/shader/shaderLoader.h"

float ParticleSystem::ticksPerSecond = 60.0f;
float ParticleSystem::secondsPerTick = 1.0f/60.0f;

ParticleSystem::ParticleSystem(unsigned int particleCount):particleCount(particleCount)
{
    particles.resize(particleCount);
}

void ParticleSystem::init(){


    for(unsigned int i=0;i<particleCount;++i){
        Particle p;
        p.position = vec4(glm::sphericalRand(15.0f),1);
        p.velocity = vec4(glm::sphericalRand(1.0f),1);
        addParticle(p);
    }

    particleBuffer.set(particles,GL_DYNAMIC_DRAW);
    particleBuffer.setDrawMode(GL_POINTS);

    initialized = true;

    particleShader = ShaderLoader::instance()->load<ParticleShader>("geometry/particles.glsl");
    deferredParticleShader = ShaderLoader::instance()->load<DeferredParticleShader>("geometry/deferred_particles.glsl");
}

void ParticleSystem::reset()
{
    if (!initialized) return;

    for(unsigned int i=0;i<particleCount;++i){
        Particle& p = particles[i];
        p = Particle();
    }

    particleBuffer.updateVertexBuffer(&particles[0],particles.size(),0);
}

void ParticleSystem::nextTick()
{
    ++tick;
}

void ParticleSystem::update()
{
    if( uploadDataNextUpdate ){
        updateParticleBuffer();
        uploadDataNextUpdate = false;
    }
}

void ParticleSystem::interpolate(float interpolation)
{
    this->interpolation = interpolation;
}


void ParticleSystem::render(Camera *cam){
    //    cout<<tick<<" ParticleSystem::renderr()"<<endl;

    if(blending){
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    }

    particleShader->bind();

    particleShader->uploadAll(model,cam->view,cam->proj);
    particleShader->uploadTexture(arrayTexture);

    particleShader->uploadTiming(tick,interpolation);
    particleShader->uploadTimestep(secondsPerTick);

//    particleBuffer.bindAndDraw();
    //draw old particles first, so new ones are on top
    particleBuffer.bind();
    particleBuffer.draw(nextParticle,particleCount-nextParticle);
    particleBuffer.draw(0,nextParticle);
    particleBuffer.unbind();

    particleShader->unbind();

    if(blending){
        glDisable(GL_BLEND);
    }
}

void ParticleSystem::renderDeferred(Camera *cam, raw_Texture* detphTexture)
{

    deferredParticleShader->bind();

    deferredParticleShader->uploadAll(model,cam->view,cam->proj);
    deferredParticleShader->uploadTexture(arrayTexture);
    deferredParticleShader->uploadDepthTexture(detphTexture);
    deferredParticleShader->uploadTiming(tick,interpolation);
    deferredParticleShader->uploadTimestep(secondsPerTick);

    deferredParticleShader->uploadCameraParameters(vec2(cam->zNear,cam->zFar));
//    particleBuffer.bindAndDraw();

    particleBuffer.bind();
    particleBuffer.draw(nextParticle,particleCount-nextParticle);
    particleBuffer.draw(0,nextParticle);
    particleBuffer.unbind();

    deferredParticleShader->unbind();

}


void ParticleSystem::addParticle(Particle &p){
    p.start = tick+1;


    particles[nextParticle] = p;
    nextParticle = (nextParticle+1)%particleCount;

    newParticles++;

}

//Particle &ParticleSystem::getNextParticle()
//{
//    Particle &p = particles[nextParticle];
//    p = Particle();
//    nextParticle = (nextParticle+1)%particleCount;
//    p.start = tick+1;

//    newParticles++;
//    return p;
//}

void ParticleSystem::updateParticleBuffer(){
    //    cout<<tick<<" ParticleSystem::updateParticleBuffer()"<<endl;

    if(newParticles>particleCount){
//        cout<<"warning: new particles spawned = "<<newParticles<<" , particle system size = "<<particleCount<<endl;
        int size = particleCount;
        int offset = 0;
        particleBuffer.updateVertexBuffer(&particles[offset],size,offset);
    }else if(nextParticle>saveParticle){
        int size = (nextParticle-saveParticle);
        int offset = saveParticle;

        particleBuffer.updateVertexBuffer(&particles[offset],size,offset);
    }else if(nextParticle<saveParticle){
        int size = (particleCount-saveParticle);
        int offset = saveParticle;
        particleBuffer.updateVertexBuffer(&particles[offset],size,offset);


        size = (nextParticle);
        offset = 0;
        particleBuffer.updateVertexBuffer(&particles[0],size,offset);
    }

    saveParticle = nextParticle;
    newParticles = 0;


}

void ParticleSystem::flush(){
    uploadDataNextUpdate = true;
}


