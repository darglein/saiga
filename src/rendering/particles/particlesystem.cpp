#include "rendering/particles/particlesystem.h"


ParticleSystem::ParticleSystem(unsigned int particleCount):particleCount(particleCount)
{
    particles.resize(particleCount);
}

void ParticleSystem::init(){


    for(unsigned int i=0;i<particleCount;++i){
        Particle p;
        p.position = glm::sphericalRand(15.0f);
        p.velocity = glm::sphericalRand(1.0f);
        p.color = vec4(glm::linearRand(vec3(0),vec3(1)),1);
        p.radius = 0.2f;
        addParticle(p);
    }

    particleBuffer.set(particles);
    particleBuffer.setDrawMode(GL_POINTS);
}

void ParticleSystem::update()
{
    tick++;

    if( uploadDataNextUpdate ){
        updateParticleBuffer();
        uploadDataNextUpdate = false;
    }
}


void ParticleSystem::render(Camera *cam, float interpolation){


    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    particleShader->bind();

    particleShader->uploadAll(model,cam->view,cam->proj);

    particleShader->uploadTiming(tick,interpolation);
    particleShader->uploadTimestep(timestep);
    particleBuffer.bindAndDraw();
    particleShader->unbind();

    glDisable(GL_BLEND);
}


void ParticleSystem::addParticle(Particle &p){
    p.start = tick;


    particles[nextParticle] = p;
    nextParticle = (nextParticle+1)%particleCount;

}

void ParticleSystem::updateParticleBuffer(){


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
        particleBuffer.updateVertexBuffer(&particles[saveParticle],size,offset);
    }

    saveParticle = nextParticle;

}

void ParticleSystem::flush(){
    uploadDataNextUpdate = true;
}


