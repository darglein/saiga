﻿/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include <iostream>
#include <vector>
#include <cuda.h>
#include <cuda_runtime.h>
#include "saiga/util/glm.h"
#include <thrust/device_vector.h>
#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/device_helper.h"


struct Particle{
    vec3 position;
    vec3 velocity;
};



__global__ static
void updateParticles(Saiga::ArrayView<Particle> particles)
{
    Saiga::CUDA::ThreadInfo<> ti;
    if(ti.thread_id >= particles.size())
        return;
    Particle& p = particles[ti.thread_id];
    p.position += p.velocity;
}

void particleSampleThrustSaiga()
{
    const int N = 100;
    const int k = 3;
    thrust::host_vector<Particle> particles(N);
    thrust::device_vector<Particle> d_particles(N);

    for(Particle& p :particles)
    {
        p.position = vec3(0);
        p.velocity = glm::linearRand(vec3(-1),vec3(1));
    }


    thrust::copy(particles.begin(),particles.end(),d_particles.begin());


    for(int i = 0; i < k; ++i)
    {
        const int BLOCK_SIZE = 128;
        updateParticles<<<Saiga::CUDA::getBlockCount(N,BLOCK_SIZE),BLOCK_SIZE>>>(d_particles);
    }

    thrust::copy(d_particles.begin(),d_particles.end(),particles.begin());


    for(Particle& p : particles)
    {
        cout << p.position << " " << p.velocity << endl;
    }
    cout << "done." << endl;
}



__global__ static
void updateParticles2(Particle* particles, int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if(tid >= N)
        return;
    Particle& p = particles[tid];
    p.position += p.velocity;

}


void particleSample()
{
    const int N = 100;
    const int k = 3;
    std::vector<Particle> particles(N);

    for(Particle& p :particles)
    {
        p.position = vec3(0);
        p.velocity = glm::linearRand(vec3(-1),vec3(1));
    }

    auto size = sizeof(Particle) * N;
    Particle* d_particles;
    cudaMalloc(&d_particles,size);
    cudaMemcpy(d_particles,particles.data(),size,cudaMemcpyHostToDevice);

    for(int i = 0; i < k; ++i)
    {
        const int BLOCK_SIZE = 128;
        int numberOfBlocks = ( N + (BLOCK_SIZE - int(1)) ) / (BLOCK_SIZE);
        updateParticles2<<<numberOfBlocks,BLOCK_SIZE>>>(d_particles,N);
    }


    cudaMemcpy(particles.data(),d_particles,size,cudaMemcpyDeviceToHost);
    cudaFree(d_particles);

    for(Particle& p : particles)
    {
        cout << p.position << " " << p.velocity << endl;
    }
    cout << "done." << endl;
}



int main(int argc, char *argv[])
{
    particleSample();
}
