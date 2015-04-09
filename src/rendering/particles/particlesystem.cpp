#include "rendering/particles/particlesystem.h"

float ParticleSystem::ticksPerSecond = 60.0f;
float ParticleSystem::secondsPerTick = 1.0f/60.0f;

ParticleSystem::ParticleSystem(unsigned int particleCount):particleCount(particleCount)
{
    particles.resize(particleCount);
}

void ParticleSystem::init(){


    for(unsigned int i=0;i<particleCount;++i){
        Particle p;
        p.position = glm::sphericalRand(15.0f);
        p.velocity = vec4(glm::sphericalRand(1.0f),1);
        addParticle(p);
    }

    particleBuffer.set(particles);
    particleBuffer.setDrawMode(GL_POINTS);
}

void ParticleSystem::update()
{
    //    cout<<tick<<" ParticleSystem::update()"<<endl;
    tick++;

    if( uploadDataNextUpdate ){
        updateParticleBuffer();
        uploadDataNextUpdate = false;
    }
}


void ParticleSystem::render(Camera *cam, float interpolation){
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
    particleBuffer.bindAndDraw();
    particleShader->unbind();

    if(blending){
        glDisable(GL_BLEND);
    }
}


void ParticleSystem::addParticle(Particle &p){
    p.start = tick+1;


    particles[nextParticle] = p;
    nextParticle = (nextParticle+1)%particleCount;

}

Particle &ParticleSystem::getNextParticle()
{
    Particle &p = particles[nextParticle];
    nextParticle = (nextParticle+1)%particleCount;
    p.start = tick+1;

    return p;
}

void ParticleSystem::updateParticleBuffer(){
    //    cout<<tick<<" ParticleSystem::updateParticleBuffer()"<<endl;

    if(nextParticle>saveParticle){
        int size = (nextParticle-saveParticle);
        int offset = saveParticle;

        particleBuffer.updateVertexBuffer(&particles[saveParticle],size,offset);
    }

    if(nextParticle<saveParticle){
        int size = (particleCount-saveParticle);
        int offset = saveParticle;
        particleBuffer.updateVertexBuffer(&particles[saveParticle],size,offset);

        size = (nextParticle);
        offset = 0;
        particleBuffer.updateVertexBuffer(&particles[0],size,offset);
    }

    saveParticle = nextParticle;



}

void ParticleSystem::flush(){
    uploadDataNextUpdate = true;
}


