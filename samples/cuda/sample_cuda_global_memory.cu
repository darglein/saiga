/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/math/math.h"
#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/device_helper.h"
#include "saiga/cuda/memory.h"
#include "saiga/cuda/tests/test_helper.h"
using namespace Saiga;
// cuobjdump globalMemory --dump-sass -arch sm_61 > sass.txt

//#define LECTURE

struct Particle
{
    vec3 position;
    float radius;
    vec3 velocity;
    float invMass;

    HD bool operator==(const Particle& other) const
    {
        return position == other.position && radius == other.radius && velocity == other.velocity &&
               invMass == other.invMass;
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


__global__ static void integrateEulerBase(Saiga::ArrayView<Particle> srcParticles,
                                          Saiga::ArrayView<Particle> dstParticles, float dt)
{
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= srcParticles.size()) return;

    Particle p = srcParticles[ti.thread_id];
    p.position += p.velocity * dt;
    p.velocity += vec3(0, -9.81, 0) * dt;
    dstParticles[ti.thread_id] = p;
}


#ifndef LECTURE

__global__ static void integrateEulerVector(Saiga::ArrayView<Particle> srcParticles,
                                            Saiga::ArrayView<Particle> dstParticles, float dt)
{
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= srcParticles.size()) return;

    Particle p;

    Saiga::CUDA::vectorCopy(srcParticles.data() + ti.thread_id, &p);
    p.position += p.velocity * dt;
    p.velocity += vec3(0, -9.81, 0) * dt;
    Saiga::CUDA::vectorCopy(&p, dstParticles.data() + ti.thread_id);
}


__global__ static void integrateEulerInverseVector(Saiga::ArrayView<PositionRadius> srcPr,
                                                   Saiga::ArrayView<VelocityMass> srcVm,
                                                   Saiga::ArrayView<PositionRadius> dstPr,
                                                   Saiga::ArrayView<VelocityMass> dstVm, float dt)
{
    Saiga::CUDA::ThreadInfo<> ti;
    if (ti.thread_id >= srcPr.size()) return;

    PositionRadius pr;
    VelocityMass vm;

    Saiga::CUDA::vectorCopy(srcPr.data() + ti.thread_id, &pr);
    Saiga::CUDA::vectorCopy(srcVm.data() + ti.thread_id, &vm);

    pr.position += vm.velocity * dt;
    vm.velocity += vec3(0, -9.81, 0) * dt;

    Saiga::CUDA::vectorCopy(&pr, dstPr.data() + ti.thread_id);
    Saiga::CUDA::vectorCopy(&vm, dstVm.data() + ti.thread_id);
}

template <unsigned int BLOCK_SIZE>
__global__ static void integrateEulerSharedVector(Saiga::ArrayView<Particle> srcParticles,
                                                  Saiga::ArrayView<Particle> dstParticles, float dt)
{
    static_assert(sizeof(Particle) % sizeof(int4) == 0, "Invalid particle size");

    const unsigned int WARPS_PER_BLOCK = BLOCK_SIZE / SAIGA_WARP_SIZE;
    __shared__ Particle tmp[WARPS_PER_BLOCK][SAIGA_WARP_SIZE];


    Saiga::CUDA::ThreadInfo<> ti;

    const auto cycles = sizeof(Particle) / sizeof(int4);
    const auto step   = SAIGA_WARP_SIZE;

    // Start offset into particle array for this warp
    auto warpStart = ti.warp_id * SAIGA_WARP_SIZE;

    // Check if complete warp is outside
    if (warpStart >= srcParticles.size()) return;


    auto begin = warpStart * cycles;
    auto end   = std::min(srcParticles.size() * cycles, begin + step * cycles);
    CUDA_ASSERT(begin < end);

    auto lbegin = 0;
    auto lend   = end - begin;

    int4* srcptr = reinterpret_cast<int4*>(srcParticles.data()) + begin;
    int4* dstptr = reinterpret_cast<int4*>(dstParticles.data()) + begin;
    int4* lptr   = reinterpret_cast<int4*>(tmp[ti.warp_lane]) + lbegin;

    for (auto i = lbegin + ti.lane_id; i < lend; i += step)
    {
        lptr[i] = srcptr[i];
    }
    __syncwarp();

    Particle& p = tmp[ti.warp_lane][ti.lane_id];
    p.position += p.velocity * dt;
    p.velocity += vec3(0, -9.81, 0) * dt;

    __syncwarp();
    for (auto i = lbegin + ti.lane_id; i < lend; i += step)
    {
        dstptr[i] = lptr[i];
    }
}

#endif

// nvcc $CPPFLAGS -I ../../../src/ -I ../../../build/include/ -ptx -gencode=arch=compute_52,code=compute_52 -g
// -std=c++11 --expt-relaxed-constexpr main.cu

void particleTest()
{
    size_t N                      = 1 * 1000 * 1000;
    size_t readWrites             = N * 2 * sizeof(Particle);
    const unsigned int BLOCK_SIZE = 128;

    thrust::host_vector<Particle> hp(N);
    for (size_t i = 0; i < N; ++i)
    {
        hp[i].position = vec3(i, i * i, i / 1242.0f);
        hp[i].velocity = vec3(-235, -i * i, i + 3465345);
        hp[i].radius   = i * 345345;
    }


    thrust::host_vector<Particle> ref, test;
    thrust::device_vector<Particle> particles  = hp;
    thrust::device_vector<Particle> particles2 = hp;

    thrust::device_vector<PositionRadius> pr(N), pr2(N);
    thrust::device_vector<VelocityMass> vm(N), vm2(N);

    int its = 50;

#ifndef LECTURE
    Saiga::CUDA::PerformanceTestHelper pth("Euler Integration", readWrites);

    {
        particles = hp;
        auto st   = Saiga::measureObject<Saiga::CUDA::ScopedTimer>(
            its, [&]() { integrateEulerBase<<<THREAD_BLOCK(N, BLOCK_SIZE)>>>(particles, particles2, 0.1f); });
        ref = particles2;
        pth.addMeassurement("integrateEulerBase", st.median);
        CUDA_SYNC_CHECK_ERROR();
    }


    {
        particles = hp;
        auto st   = Saiga::measureObject<Saiga::CUDA::ScopedTimer>(
            its, [&]() { integrateEulerVector<<<THREAD_BLOCK(N, BLOCK_SIZE)>>>(particles, particles2, 0.1f); });
        test = particles2;
        SAIGA_ASSERT(test == ref);
        pth.addMeassurement("integrateEulerVector", st.median);
        CUDA_SYNC_CHECK_ERROR();
    }


    {
        particles = hp;
        auto st   = Saiga::measureObject<Saiga::CUDA::ScopedTimer>(
            its, [&]() { integrateEulerInverseVector<<<THREAD_BLOCK(N, BLOCK_SIZE)>>>(pr, vm, pr2, vm2, 0.1f); });
        pth.addMeassurement("integrateEulerInverseVector", st.median);
        CUDA_SYNC_CHECK_ERROR();
    }


    {
        particles = hp;
        auto st   = Saiga::measureObject<Saiga::CUDA::ScopedTimer>(its, [&]() {
            integrateEulerSharedVector<BLOCK_SIZE><<<THREAD_BLOCK(N, BLOCK_SIZE)>>>(particles, particles2, 0.1f);
        });
        test      = particles2;
        SAIGA_ASSERT(test == ref);
        pth.addMeassurement("integrateEulerSharedVector", st.median);
    }


    {
        auto st = Saiga::measureObject<Saiga::CUDA::ScopedTimer>(its, [&]() {
            cudaMemcpy(particles2.data().get(), particles.data().get(), N * sizeof(Particle), cudaMemcpyDeviceToDevice);
        });
        pth.addMeassurement("cudaMemcpy", st.median);
    }
#endif
    CUDA_SYNC_CHECK_ERROR();
}

// cuobjdump ./globalMemory -arch=sm_30 -sass -ptx > test.txt


void memcpyTest()
{
    size_t N          = 100 * 1000 * 1000;
    size_t readWrites = N * 2 * sizeof(int);

    thrust::device_vector<int> src(N);
    thrust::device_vector<int> dest(N);


#ifndef LECTURE
    // Only a single execution
    // This might not be so accurate
    {
        float t = 0;
        {
            Saiga::CUDA::ScopedTimer timer(t);
            cudaMemcpy(dest.data().get(), src.data().get(), N * sizeof(int), cudaMemcpyDeviceToDevice);
        }
        double bandwidth = readWrites / t / (1000 * 1000);
        std::cout << "Time: " << t << " ms,   Bandwidth: " << bandwidth << " GB/s" << std::endl;
    }

    Saiga::CUDA::PerformanceTestHelper pth("Memcpy", readWrites);
    // Test 10 times and use the median time
    int its = 50;
    {
        auto st = Saiga::measureObject<Saiga::CUDA::ScopedTimer>(its, [&]() {
            cudaMemcpy(thrust::raw_pointer_cast(dest.data()), thrust::raw_pointer_cast(src.data()), N * sizeof(int),
                       cudaMemcpyDeviceToDevice);
        });
        pth.addMeassurement("cudaMemcpy", st.median);
    }
#endif


    CUDA_SYNC_CHECK_ERROR();
}

int main(int argc, char* argv[])
{
    memcpyTest();
    particleTest();
    return 0;
}
