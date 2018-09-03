/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/util/glm.h"
#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/device_helper.h"
#include "saiga/cuda/memory.h"
#include "saiga/cuda/tests/test_helper.h"


struct Particle
{
    vec3 position;
    float radius;
    vec3 velocity;
    float invMass;
};

struct PositionRadius
{
    vec3 position;
    float radius;
};

struct VelocityMass
{
    vec3 velocity;
    float invMass;
};

__global__ static
void integrateEuler1(Saiga::ArrayView<Particle> particles, float dt)
{
    Saiga::CUDA::ThreadInfo<> ti;
    if(ti.thread_id >= particles.size())
        return;
    Particle& p = particles[ti.thread_id];
    p.position += p.velocity * dt;
    p.velocity += vec3(0,-9.81,0) * dt;
}


__global__ static
void integrateEuler2(Saiga::ArrayView<Particle> particles, float dt)
{
    Saiga::CUDA::ThreadInfo<> ti;
    if(ti.thread_id >= particles.size())
        return;
    Particle p = particles[ti.thread_id];
    p.position += p.velocity * dt;
    p.velocity += vec3(0,-9.81,0) * dt;
    particles[ti.thread_id] = p;
}

__global__ static
void integrateEuler3(Saiga::ArrayView<PositionRadius> prs, Saiga::ArrayView<VelocityMass> vms, float dt)
{
    Saiga::CUDA::ThreadInfo<> ti;
    if(ti.thread_id >= prs.size())
        return;
    PositionRadius& pr = prs[ti.thread_id];
    VelocityMass& vm = vms[ti.thread_id];

    pr.position += vm.velocity * dt;
    vm.velocity += vec3(0,-9.81,0) * dt;
}

__global__ static
void integrateEuler4(Saiga::ArrayView<PositionRadius> prs, Saiga::ArrayView<VelocityMass> vms, float dt)
{
    Saiga::CUDA::ThreadInfo<> ti;
    if(ti.thread_id >= prs.size())
        return;
    PositionRadius pr = prs[ti.thread_id];
    VelocityMass vm   = vms[ti.thread_id];

    pr.position += vm.velocity * dt;
    vm.velocity += vec3(0,-9.81,0) * dt;

    prs[ti.thread_id] = pr;
    vms[ti.thread_id] = vm;
}

__global__ static
void integrateEuler5(Saiga::ArrayView<PositionRadius> prs, Saiga::ArrayView<VelocityMass> vms, float dt)
{
    Saiga::CUDA::ThreadInfo<> ti;
    if(ti.thread_id >= prs.size())
        return;

    PositionRadius pr;
    reinterpret_cast<int4*>(&pr)[0] = reinterpret_cast<int4*>(prs.data()+ti.thread_id)[0];
    VelocityMass vm;
    reinterpret_cast<int4*>(&vm)[0] = reinterpret_cast<int4*>(vms.data()+ti.thread_id)[0];


    pr.position += vm.velocity * dt;
    vm.velocity += vec3(0,-9.81,0) * dt;

    reinterpret_cast<int4*>(prs.data()+ti.thread_id)[0] = reinterpret_cast<int4*>(&pr)[0];
    reinterpret_cast<int4*>(vms.data()+ti.thread_id)[0] = reinterpret_cast<int4*>(&vm)[0];
}

template<unsigned int BLOCK_SIZE>
__global__ static
void integrateEuler6(Saiga::ArrayView<Particle> particles, float dt)
{
    __shared__ Particle tmp[BLOCK_SIZE];

    Saiga::CUDA::ThreadInfo<> ti;

    auto start = ti.warp_id * 32;
    auto start4 = ti.warp_id * 32 * 2;

    auto offset1 = ti.lane_id;
    auto offset2 = ti.lane_id + 32;

    auto end4 = particles.size() * 2;

    if(start4 + offset1 < end4)
        reinterpret_cast<int4*>(tmp)[offset1] = reinterpret_cast<int4*>(particles.data()+start)[offset1];
    if(start4 + offset2 < end4)
        reinterpret_cast<int4*>(tmp)[offset2] = reinterpret_cast<int4*>(particles.data()+start)[offset2];

    Particle& p = tmp[ti.lane_id];
    p.position += p.velocity * dt;
    p.velocity += vec3(0,-9.81,0) * dt;

    if(start4 + offset1 < end4)
        reinterpret_cast<int4*>(particles.data()+start)[offset1]    = reinterpret_cast<int4*>(tmp)[offset1]    ;
    if(start4 + offset2 < end4)
        reinterpret_cast<int4*>(particles.data()+start)[offset2] = reinterpret_cast<int4*>(tmp)[offset2] ;
}


void particleTest()
{
    size_t N = 10 * 1000 * 1000;
    size_t readWrites = N * 2 * sizeof(Particle);
    const unsigned int BLOCK_SIZE = 128;

    thrust::device_vector<Particle> particles(N);
    thrust::device_vector<PositionRadius> pr(N);
    thrust::device_vector<VelocityMass> vm(N);

    int its = 50;

    Saiga::CUDA::PerformanceTestHelper pth("Euler Integration", readWrites);

    {
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
            integrateEuler1<<<THREAD_BLOCK(N,BLOCK_SIZE)>>>(particles,0.1f);
        });
        pth.addMeassurement("integrateEuler1",st.median);
    }
    {
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
            integrateEuler2<<<THREAD_BLOCK(N,BLOCK_SIZE)>>>(particles,0.1f);
        });
        pth.addMeassurement("integrateEuler2",st.median);
    }

    {
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
            integrateEuler3<<<THREAD_BLOCK(N,BLOCK_SIZE)>>>(pr,vm,0.1f);
        });
        pth.addMeassurement("integrateEuler3",st.median);
    }

    {
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
            integrateEuler4<<<THREAD_BLOCK(N,BLOCK_SIZE)>>>(pr,vm,0.1f);
        });
        pth.addMeassurement("integrateEuler4",st.median);
    }

    {
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
            integrateEuler5<<<THREAD_BLOCK(N,BLOCK_SIZE)>>>(pr,vm,0.1f);
        });
        pth.addMeassurement("integrateEuler5",st.median);
    }

    {
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
            integrateEuler6<BLOCK_SIZE><<<THREAD_BLOCK(N,BLOCK_SIZE)>>>(particles,0.1f);
        });
        pth.addMeassurement("integrateEuler6",st.median);
    }

    CUDA_SYNC_CHECK_ERROR();
}

void memcpyTest()
{
    size_t N = 100 * 1000 * 1000;
    size_t readWrites = N * 2 * sizeof(int);

    thrust::device_vector<int> src(N);
    thrust::device_vector<int> dest(N);


    // Only a single execution
    // This might not be so accurate
    {
        float t  = 0;
        {
            Saiga::CUDA::CudaScopedTimer timer(t);
            cudaMemcpy(dest.data().get(),src.data().get(),N * sizeof(int),cudaMemcpyDeviceToDevice);
        }
        double bandwidth = readWrites / t / (1000 * 1000);
        cout << "Time: " << t << " ms,   Bandwidth: " << bandwidth << " GB/s" << endl;
    }

    Saiga::CUDA::PerformanceTestHelper pth("Memcpy", readWrites);
    // Test 10 times and use the median time
    int its = 50;
    {
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
            cudaMemcpy(thrust::raw_pointer_cast(dest.data()),thrust::raw_pointer_cast(src.data()),N * sizeof(int),cudaMemcpyDeviceToDevice);
        });
        pth.addMeassurement("cudaMemcpy",st.min);
    }



    CUDA_SYNC_CHECK_ERROR();
}

int main(int argc, char *argv[])
{

    memcpyTest();
    particleTest();
    return 0;
}

