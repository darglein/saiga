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


struct GLM_ALIGN(32) Particle
{
    vec3 position;
    float radius;
    vec3 velocity;
    float invMass;

    HD bool operator==(const Particle& other) const
    {
        return position == other.position
                && radius == other.radius
                && velocity == other.velocity
                && invMass == other.invMass;
    }
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
void particleCopy(Saiga::ArrayView<Particle> particles1, Saiga::ArrayView<Particle> particles2)
{
    Saiga::CUDA::ThreadInfo<> ti;
//    if(ti.thread_id >= particles1.size())
//        return;

    auto size4 = particles1.size() * 2;


    for(auto tid = ti.thread_id; tid < size4; tid += ti.grid_size)
    {
        int4* src = reinterpret_cast<int4*>(particles1.data());
        int4* dst = reinterpret_cast<int4*>(particles2.data());
        dst[tid] = src[tid];
    }

}


__global__ static
void integrateEuler1(Saiga::ArrayView<Particle> srcParticles, Saiga::ArrayView<Particle> dstParticles, float dt)
{
    Saiga::CUDA::ThreadInfo<> ti;
    if(ti.thread_id >= srcParticles.size()) return;
    Particle p = srcParticles[ti.thread_id];
    p.position += p.velocity * dt;
    p.velocity += vec3(0,-9.81,0) * dt;
    dstParticles[ti.thread_id] = p;
}


__global__ static
void integrateEuler2(Saiga::ArrayView<Particle> srcParticles, Saiga::ArrayView<Particle> dstParticles, float dt)
{
    Saiga::CUDA::ThreadInfo<> ti;
    if(ti.thread_id >= srcParticles.size())
        return;
    Particle p;
    Saiga::CUDA::vectorCopy(srcParticles.data() + ti.thread_id,&p);
    p.position += p.velocity * dt;
    p.velocity += vec3(0,-9.81,0) * dt;
    Saiga::CUDA::vectorCopy(&p,dstParticles.data() + ti.thread_id);
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
void integrateEuler4(
        Saiga::ArrayView<PositionRadius> srcPr, Saiga::ArrayView<VelocityMass> srcVm,
        Saiga::ArrayView<PositionRadius> dstPr, Saiga::ArrayView<VelocityMass> dstVm, float dt)
{
    Saiga::CUDA::ThreadInfo<> ti;
    if(ti.thread_id >= srcPr.size()) return;

    PositionRadius pr;
    VelocityMass vm;
    Saiga::CUDA::vectorCopy(srcPr.data()+ti.thread_id,&pr);
    Saiga::CUDA::vectorCopy(srcVm.data()+ti.thread_id,&vm);

    pr.position += vm.velocity * dt;
    vm.velocity += vec3(0,-9.81,0) * dt;

    Saiga::CUDA::vectorCopy(&pr,dstPr.data()+ti.thread_id);
    Saiga::CUDA::vectorCopy(&vm,dstVm.data()+ti.thread_id);
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
void integrateEuler6(Saiga::ArrayView<Particle> srcParticles, Saiga::ArrayView<Particle> dstParticles, float dt)
{
    const unsigned int WARPS_PER_BLOCK = BLOCK_SIZE / 32;
    __shared__ Particle tmp[WARPS_PER_BLOCK][32];

    Saiga::CUDA::ThreadInfo<> ti;

    auto start = ti.warp_id * 32;
    auto start4 = ti.warp_id * 32 * 2;

    auto offset1 = ti.lane_id;
    auto offset2 = ti.lane_id + 32;

    auto end4 = srcParticles.size() * 2;

    if(start4 + offset1 < end4)
        reinterpret_cast<int4*>(tmp[ti.warp_lane])[offset1] = reinterpret_cast<int4*>(srcParticles.data()+start)[offset1];
    if(start4 + offset2 < end4)
        reinterpret_cast<int4*>(tmp[ti.warp_lane])[offset2] = reinterpret_cast<int4*>(srcParticles.data()+start)[offset2];

    Particle& p = tmp[ti.warp_lane][ti.lane_id];
    p.position += p.velocity * dt;
    p.velocity += vec3(0,-9.81,0) * dt;

    if(start4 + offset1 < end4)
        reinterpret_cast<int4*>(dstParticles.data()+start)[offset1]    = reinterpret_cast<int4*>(tmp[ti.warp_lane])[offset1]    ;
    if(start4 + offset2 < end4)
        reinterpret_cast<int4*>(dstParticles.data()+start)[offset2] = reinterpret_cast<int4*>(tmp[ti.warp_lane])[offset2] ;
}

//nvcc $CPPFLAGS -I ../../../src/ -I ../../../build/include/ -ptx -gencode=arch=compute_52,code=compute_52 -g -std=c++11 --expt-relaxed-constexpr main.cu

void particleTest()
{
    size_t N = 1 * 1000 * 1000;
    size_t readWrites = N * 2 * sizeof(Particle);
    const unsigned int BLOCK_SIZE = 128;

    thrust::host_vector<Particle> hp(N);
    for(size_t i = 0; i < N; ++i)
    {
        hp[i].position = vec3(i,i*i,i / 1242.0f);
        hp[i].velocity = vec3(-235,-i*i,i + 3465345);
        hp[i].radius = i*345345;
    }


    thrust::host_vector<Particle> ref, test;
    thrust::device_vector<Particle> particles = hp;
    thrust::device_vector<Particle> particles2 = hp;

    thrust::device_vector<PositionRadius> pr(N), pr2(N);
    thrust::device_vector<VelocityMass> vm(N), vm2(N);

    int its = 50;

    Saiga::CUDA::PerformanceTestHelper pth("Euler Integration", readWrites);

    {
        particles = hp;
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
            integrateEuler1<<<THREAD_BLOCK(N,BLOCK_SIZE)>>>(particles,particles2,0.1f);
        });
        ref = particles2;
        pth.addMeassurement("integrateEuler1",st.median);
        CUDA_SYNC_CHECK_ERROR();
    }
    {
         particles = hp;
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
            integrateEuler2<<<THREAD_BLOCK(N,BLOCK_SIZE)>>>(particles,particles2,0.1f);
        });
        test = particles2;
        SAIGA_ASSERT(test == ref);
        pth.addMeassurement("integrateEuler2",st.median);
        CUDA_SYNC_CHECK_ERROR();
    }

    {
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
            integrateEuler3<<<THREAD_BLOCK(N,BLOCK_SIZE)>>>(pr,vm,0.1f);
        });
        pth.addMeassurement("integrateEuler3",st.median);
        CUDA_SYNC_CHECK_ERROR();
    }

    {
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
            integrateEuler4<<<THREAD_BLOCK(N,BLOCK_SIZE)>>>(pr,vm,pr2,vm2,0.1f);
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
          particles = hp;
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
            integrateEuler6<BLOCK_SIZE><<<THREAD_BLOCK(N,BLOCK_SIZE)>>>(particles,particles2,0.1f);
        });
        test = particles2;
        SAIGA_ASSERT(test == ref);
        pth.addMeassurement("integrateEuler6",st.median);
    }

    {
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
            particleCopy<<<THREAD_BLOCK(N,BLOCK_SIZE)>>>(particles,particles2);
        });
        pth.addMeassurement("particleCopy",st.median);
    }

    CUDA_SYNC_CHECK_ERROR();
}

//cuobjdump ./globalMemory -arch=sm_30 -sass -ptx > test.txt


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
        pth.addMeassurement("cudaMemcpy",st.median);
    }



    CUDA_SYNC_CHECK_ERROR();
}

int main(int argc, char *argv[])
{

    memcpyTest();
    particleTest();
    return 0;
}

