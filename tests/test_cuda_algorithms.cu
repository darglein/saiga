/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/math/random.h"
#include "saiga/cuda/CudaInfo.h"
#include "saiga/cuda/bitonicSort.h"
#include "saiga/cuda/device_helper.h"
#include "saiga/cuda/dot.h"
#include "saiga/cuda/event.h"
#include "saiga/cuda/math/inverse.h"
#include "saiga/cuda/pinned_vector.h"
#include "saiga/cuda/random.h"
#include "saiga/cuda/reduce.h"
#include "saiga/cuda/reduce_global.h"
#include "saiga/cuda/scan.h"
#include "saiga/cuda/shuffle_copy.h"
#include "saiga/cuda/stream.h"
#include "saiga/cuda/thrust_helper.h"
#include "saiga/cuda/thrust_tmp_allocator.h"

#include "gtest/gtest.h"

#include <algorithm>
#include <climits>
#include <numeric>
#include <thrust/mr/allocator.h>
#include <thrust/sort.h>

#include <thrust/execution_policy.h>

namespace Saiga
{
template <typename T>
static std::vector<T> RandomVector(int n, int min_value, int max_value)
{
    std::vector<T> v(n);
    for (auto& x : v) x = T(Random::uniformInt(min_value, max_value));
    return v;
}

template <typename T>
static std::vector<T> ToHost(const thrust::device_vector<T>& d)
{
    thrust::host_vector<T> h = d;
    return std::vector<T>(h.begin(), h.end());
}

// ======================== Reductions ========================

template <typename T, unsigned int BLOCK_SIZE>
static void TestGlobalReduce(int n)
{
    auto h_data = RandomVector<T>(n, -100, 100);
    T expected  = std::accumulate(h_data.begin(), h_data.end(), T(0));
    thrust::device_vector<T> d_data(h_data);
    thrust::device_vector<T> d_out(1);

    // Use few blocks so that the grid stride loops are exercised.
    int blocks = std::max(1, std::min(n / int(BLOCK_SIZE), 7));

    d_out[0] = 0;
    CUDA::reduceBlockShared<T, BLOCK_SIZE><<<blocks, BLOCK_SIZE>>>(d_data, d_out.data().get());
    EXPECT_EQ(T(d_out[0]), expected) << "reduceBlockShared n=" << n << " block=" << BLOCK_SIZE;

    d_out[0] = 0;
    CUDA::reduceBlockSharedAtomic<T, BLOCK_SIZE><<<blocks, BLOCK_SIZE>>>(d_data, d_out.data().get());
    EXPECT_EQ(T(d_out[0]), expected) << "reduceBlockSharedAtomic n=" << n << " block=" << BLOCK_SIZE;

    d_out[0] = 0;
    CUDA::reduceAtomic<T, BLOCK_SIZE><<<blocks, BLOCK_SIZE>>>(d_data, d_out.data().get());
    EXPECT_EQ(T(d_out[0]), expected) << "reduceAtomic n=" << n << " block=" << BLOCK_SIZE;

    auto h_data2   = RandomVector<T>(n, -10, 10);
    T expected_dot = std::inner_product(h_data.begin(), h_data.end(), h_data2.begin(), T(0));
    thrust::device_vector<T> d_data2(h_data2);
    d_out[0] = 0;
    CUDA::dot<T, BLOCK_SIZE><<<blocks, BLOCK_SIZE>>>(d_data, d_data2, d_out.data().get());
    EXPECT_EQ(T(d_out[0]), expected_dot) << "dot n=" << n << " block=" << BLOCK_SIZE;
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
}

TEST(CudaReduce, GlobalSum)
{
    for (int n : {1, 2, 31, 1000, 4097, 100001})
    {
        TestGlobalReduce<int, 32>(n);
        TestGlobalReduce<int, 128>(n);
        TestGlobalReduce<int, 256>(n);
        TestGlobalReduce<int, 1024>(n);
        TestGlobalReduce<double, 128>(n);
    }
}

// Regression test: blockReduceSum reduced the per-warp partial sums with a sub-warp reduction of width
// BLOCK_SIZE/32. For a block size whose warp count is not a power of two (e.g. 96 = 3 warps) the
// partial sum of the last warp was dropped.
TEST(CudaReduce, GlobalSumNonPowerOfTwoWarps)
{
    for (int n : {96, 1000, 100001})
    {
        TestGlobalReduce<int, 96>(n);
        TestGlobalReduce<int, 192>(n);
        TestGlobalReduce<int, 320>(n);
    }
}

// Each block reduces its BLOCK_SIZE elements twice with the same shared memory.
template <unsigned int BLOCK_SIZE>
__global__ static void BlockReduceTwiceKernel(ArrayView<int> in, ArrayView<int> out)
{
    __shared__ int shared[BLOCK_SIZE / SAIGA_WARP_SIZE];
    __shared__ int shared_atomic[1];
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int v = i < in.size() ? in[i] : 0;

    int s1 = CUDA::blockReduceSum<int, BLOCK_SIZE>(v, shared, 0);
    int s2 = CUDA::blockReduceSum<int, BLOCK_SIZE>(2 * v + 1, shared, 0);
    int a1 = CUDA::blockReduceAtomicSum<int, BLOCK_SIZE>(v, shared_atomic);
    int a2 = CUDA::blockReduceAtomicSum<int, BLOCK_SIZE>(3 * v, shared_atomic);
    if (threadIdx.x == 0)
    {
        out[blockIdx.x * 4 + 0] = s1;
        out[blockIdx.x * 4 + 1] = s2;
        out[blockIdx.x * 4 + 2] = a1;
        out[blockIdx.x * 4 + 3] = a2;
    }
}

// Regression test: blockReduceSum and blockReduceAtomicSum had no barrier before writing to 'shared'. In a second call,
// the other warps could overwrite the per-warp sums while warp 0 was still reading the values of the first call (found
// by compute-sanitizer --tool racecheck).
TEST(CudaReduce, BlockReduceTwice)
{
    const unsigned int BLOCK_SIZE = 1024;
    const int blocks              = 200;
    const int n                   = blocks * BLOCK_SIZE - 17;
    auto h_data                   = RandomVector<int>(n, -100, 100);
    thrust::device_vector<int> d_data(h_data);
    thrust::device_vector<int> d_out(blocks * 4);
    BlockReduceTwiceKernel<BLOCK_SIZE><<<blocks, BLOCK_SIZE>>>(d_data, d_out);
    auto h_out = ToHost(d_out);
    for (int b = 0; b < blocks; ++b)
    {
        int begin = b * BLOCK_SIZE;
        int end   = std::min(n, begin + int(BLOCK_SIZE));
        int sum   = std::accumulate(h_data.begin() + begin, h_data.begin() + end, 0);
        EXPECT_EQ(h_out[b * 4 + 0], sum) << "block " << b;
        EXPECT_EQ(h_out[b * 4 + 1], 2 * sum + int(BLOCK_SIZE)) << "block " << b;
        EXPECT_EQ(h_out[b * 4 + 2], sum) << "block " << b;
        EXPECT_EQ(h_out[b * 4 + 3], 3 * sum) << "block " << b;
    }
}

struct MaxOp
{
    __device__ int operator()(int a, int b) const { return a > b ? a : b; }
};

// Each block reduces BLOCK_SIZE elements twice (to check that consecutive calls at the same call site work).
template <int BLOCK_SIZE>
__global__ static void GenericReduceKernel(ArrayView<int> in, ArrayView<int> out_max, ArrayView<int> out_sum)
{
    int i = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    int v = i < in.size() ? in[i] : INT_MIN;
    int m = CUDA::reduce<BLOCK_SIZE>(v, MaxOp(), INT_MIN);
    if (threadIdx.x == 0) out_max[blockIdx.x] = m;

    int s = CUDA::reduce<BLOCK_SIZE>(i < in.size() ? in[i] : 0, [](int a, int b) { return a + b; }, 0);
    if (threadIdx.x == 0) out_sum[blockIdx.x] = s;
}

template <int BLOCK_SIZE>
static void TestGenericReduce(int n)
{
    auto h_data = RandomVector<int>(n, -1000, 1000);
    thrust::device_vector<int> d_data(h_data);
    int blocks = iDivUp(n, BLOCK_SIZE);
    thrust::device_vector<int> d_max(blocks), d_sum(blocks);
    GenericReduceKernel<BLOCK_SIZE><<<blocks, BLOCK_SIZE>>>(d_data, d_max, d_sum);
    auto h_max = ToHost(d_max);
    auto h_sum = ToHost(d_sum);
    for (int b = 0; b < blocks; ++b)
    {
        int begin = b * BLOCK_SIZE;
        int end   = std::min(n, begin + BLOCK_SIZE);
        EXPECT_EQ(h_max[b], *std::max_element(h_data.begin() + begin, h_data.begin() + end))
            << "block size " << BLOCK_SIZE << " block " << b;
        EXPECT_EQ(h_sum[b], std::accumulate(h_data.begin() + begin, h_data.begin() + end, 0))
            << "block size " << BLOCK_SIZE << " block " << b;
    }
}

TEST(CudaReduce, Generic)
{
    for (int n : {1, 100, 5000})
    {
        TestGenericReduce<1>(n);
        TestGenericReduce<2>(n);
        TestGenericReduce<8>(n);
        TestGenericReduce<16>(n);
        TestGenericReduce<32>(n);
        TestGenericReduce<64>(n);
        TestGenericReduce<96>(n);
        TestGenericReduce<256>(n);
        TestGenericReduce<1024>(n);
    }
}

// ======================== Scans ========================

template <unsigned int BLOCK_SIZE, int TILES>
__global__ static void BlockScanKernel(ArrayView<int> in, ArrayView<int> out)
{
    __shared__ int shared[BLOCK_SIZE / SAIGA_WARP_SIZE];
    __shared__ int prefix;
    if (threadIdx.x == 0) prefix = 0;
    __syncthreads();

    for (int t = 0; t < TILES; ++t)
    {
        int i = (blockIdx.x * TILES + t) * BLOCK_SIZE + threadIdx.x;
        int v = i < in.size() ? in[i] : 0;
        v     = CUDA::blockInclusiveScan<BLOCK_SIZE>(v, shared, &prefix);
        if (i < in.size()) out[i] = v;
    }
}

template <unsigned int BLOCK_SIZE, int TILES>
static void TestBlockScan(int n)
{
    auto h_data = RandomVector<int>(n, 0, 10);
    thrust::device_vector<int> d_data(h_data);
    thrust::device_vector<int> d_out(n, -1);
    int blocks = iDivUp(n, BLOCK_SIZE * TILES);
    BlockScanKernel<BLOCK_SIZE, TILES><<<blocks, BLOCK_SIZE>>>(d_data, d_out);
    auto h_out = ToHost(d_out);

    int errors = 0;
    for (int b = 0; b < blocks; ++b)
    {
        int sum = 0;
        for (int i = b * BLOCK_SIZE * TILES; i < std::min(n, (b + 1) * int(BLOCK_SIZE) * TILES); ++i)
        {
            sum += h_data[i];
            if (h_out[i] != sum) errors++;
        }
    }
    EXPECT_EQ(errors, 0) << "block scan with block size " << BLOCK_SIZE << " n " << n;
}

// Regression test: the scan over the per-warp sums used a shuffle of width BLOCK_SIZE/32 (invalid if that is
// not a power of two) and was executed by only BLOCK_SIZE/32 lanes with a full warp mask, which hung the kernel.
TEST(CudaScan, BlockInclusiveScan)
{
    for (int n : {1, 100, 1000, 12345})
    {
        TestBlockScan<32, 1>(n);
        TestBlockScan<64, 2>(n);
        TestBlockScan<96, 2>(n);
        TestBlockScan<128, 3>(n);
        TestBlockScan<256, 2>(n);
        TestBlockScan<1024, 1>(n);
    }
}

template <bool EXCLUSIVE>
static void TestTiledScan(int n, int max_value = 3)
{
    const unsigned int BLOCK_SIZE = 256;
    const unsigned int TILES      = 8;
    const unsigned int ELEMENTS   = BLOCK_SIZE * TILES * 4;

    auto h_data = RandomVector<unsigned int>(n, 0, max_value);
    std::vector<unsigned int> expected(n);
    unsigned int sum = 0;
    for (int i = 0; i < n; ++i)
    {
        if (EXCLUSIVE) expected[i] = sum;
        sum += h_data[i];
        if (!EXCLUSIVE) expected[i] = sum;
    }

    thrust::device_vector<unsigned int> d_data(h_data);
    thrust::device_vector<unsigned int> d_out(n, 0xFFFFFFFF);
    int blocks = CUDA::getBlockCount(n, ELEMENTS);
    thrust::device_vector<unsigned long long> d_aggregate(blocks + 1, 0);

    float ms;
    {
        CUDA::ScopedTimer timer(ms);
        CUDA::tiledSinglePassScan<EXCLUSIVE, BLOCK_SIZE, TILES, int4, true>
            <<<blocks, BLOCK_SIZE>>>(d_data, d_out, d_aggregate);
    }
    // Regression test: the decoupled look-back spun on a plain (L1 cached) load of the predecessor's aggregate. It
    // took seconds instead of microseconds to see the update, or never saw it and hung.
    EXPECT_LT(ms, 1000) << "n " << n;
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    auto h_out = ToHost(d_out);

    int errors = 0;
    for (int i = 0; i < n; ++i)
    {
        if (h_out[i] != expected[i])
        {
            if (errors < 5) ADD_FAILURE() << "index " << i << " expected " << expected[i] << " got " << h_out[i];
            errors++;
        }
    }
    EXPECT_EQ(errors, 0) << "n " << n << " exclusive " << EXCLUSIVE;
}

TEST(CudaScan, TiledSinglePassScan)
{
    for (int n : {1, 3, 4, 5, 8191, 8192, 8193, 100000, 1000003, 10000000})
    {
        TestTiledScan<true>(n);
        TestTiledScan<false>(n);
    }
}

// Regression test: the look-back stored the flag in the upper 2 bits and the sum in the lower 30 bits of a 32 bit
// status word (block 0 did not even mask its sum). Prefix sums of 2^30 and more were corrupted.
TEST(CudaScan, TiledSinglePassScanLargeSums)
{
    // The total sums are slightly above 2^30, 2^31 and 2^32 (wraps around).
    for (int max_value : {2200, 4400, 9000})
    {
        TestTiledScan<true>(1000003, max_value);
        TestTiledScan<false>(1000003, max_value);
    }
}

// ======================== Warp sort / shuffle ========================

template <unsigned int SIZE>
__global__ static void BitonicKernel(ArrayView<int> data)
{
    int i   = blockIdx.x * blockDim.x + threadIdx.x;
    int v   = data[i];
    v       = CUDA::bitonicWarpSort<int, SIZE>(v, threadIdx.x);
    data[i] = v;
}

template <unsigned int SIZE>
static void TestBitonic()
{
    int n       = 32 * 64;
    auto h_data = RandomVector<int>(n, -1000, 1000);
    thrust::device_vector<int> d_data(h_data);
    BitonicKernel<SIZE><<<n / 128, 128>>>(d_data);
    auto h_out = ToHost(d_data);
    for (int i = 0; i < n; i += SIZE)
    {
        std::sort(h_data.begin() + i, h_data.begin() + i + SIZE);
    }
    EXPECT_EQ(h_out, h_data) << "bitonic size " << SIZE;
}

TEST(CudaWarp, BitonicSort)
{
    TestBitonic<2>();
    TestBitonic<4>();
    TestBitonic<8>();
    TestBitonic<16>();
    TestBitonic<32>();
}

struct alignas(16) ShuffleParticle
{
    float4 a, b, c;
};

template <int G>
__global__ static void ShuffleCopyKernel(ArrayView<ShuffleParticle> particles)
{
    CUDA::ThreadInfo<0, G> ti;
    ShuffleParticle p;
    CUDA::loadShuffleStruct<G, ShuffleParticle, int4>(particles.data(), &p, ti.lane_id, ti.warp_id, particles.size());
    p.a.x += 1;
    p.b.y += 2;
    p.c.w += 3;
    CUDA::storeShuffleStruct<G, ShuffleParticle, int4>(particles.data(), &p, ti.lane_id, ti.warp_id, particles.size());
}

template <int G>
static void TestShuffleCopy()
{
    int n = 32 * 8 / G * G;
    std::vector<ShuffleParticle> h(n);
    for (int i = 0; i < n; ++i)
    {
        float f = i * 12;
        h[i].a  = make_float4(f, f + 1, f + 2, f + 3);
        h[i].b  = make_float4(f + 4, f + 5, f + 6, f + 7);
        h[i].c  = make_float4(f + 8, f + 9, f + 10, f + 11);
    }
    thrust::device_vector<ShuffleParticle> d(h.begin(), h.end());
    ShuffleCopyKernel<G><<<n / G * G / 64, 64>>>(d);
    thrust::host_vector<ShuffleParticle> r = d;
    for (int i = 0; i < n; ++i)
    {
        float f = i * 12;
        EXPECT_EQ(r[i].a.x, f + 1);
        EXPECT_EQ(r[i].a.y, f + 1);
        EXPECT_EQ(r[i].b.y, f + 5 + 2);
        EXPECT_EQ(r[i].c.z, f + 10);
        EXPECT_EQ(r[i].c.w, f + 11 + 3);
    }
}

TEST(CudaWarp, ShuffleCopy)
{
    TestShuffleCopy<1>();
    TestShuffleCopy<2>();
    TestShuffleCopy<4>();
}

// ======================== Random ========================

template <int N>
__global__ static void DrawRandom(ArrayView<curandState> states, ArrayView<float> out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= states.size()) return;
    curandState s = states[tid];
    for (int i = 0; i < N; ++i)
    {
        out[tid * N + i] = CUDA::linearRand(2, 5, s);
    }
    states[tid] = s;
}

// Regression test: initRandom gave thread t the sequence (t % 997) at offset (t / 997) * 10. Therefore the 11th
// number of thread t was identical to the 1st number of thread t + 997.
TEST(CudaRandom, IndependentSequences)
{
    const int n_states = 3000;
    const int N        = 32;
    thrust::device_vector<curandState> states(n_states);
    CUDA::initRandom(states, 3654);
    thrust::device_vector<float> d_out(n_states * N);
    DrawRandom<N><<<iDivUp(n_states, 128), 128>>>(states, d_out);
    auto h = ToHost(d_out);

    for (auto f : h)
    {
        EXPECT_GE(f, 2);
        EXPECT_LE(f, 5);
    }
    double mean = std::accumulate(h.begin(), h.end(), 0.0) / h.size();
    EXPECT_NEAR(mean, 3.5, 0.05);

    // Count how many (thread, index) pairs reappear in another thread.
    std::vector<std::vector<float>> sequences(n_states);
    for (int t = 0; t < n_states; ++t) sequences[t] = std::vector<float>(h.begin() + t * N, h.begin() + (t + 1) * N);

    int duplicated_sequences = 0;
    for (int t = 0; t + 997 < n_states; ++t)
    {
        auto& a = sequences[t];
        auto& b = sequences[t + 997];
        if (std::equal(a.begin() + 10, a.end(), b.begin())) duplicated_sequences++;
    }
    EXPECT_EQ(duplicated_sequences, 0);

    // The same seed must give the same numbers.
    thrust::device_vector<curandState> states2(n_states);
    CUDA::initRandom(states2, 3654);
    thrust::device_vector<float> d_out2(n_states * N);
    DrawRandom<N><<<iDivUp(n_states, 128), 128>>>(states2, d_out2);
    EXPECT_EQ(ToHost(d_out2), h);
}

__global__ static void SampleConeKernel(ArrayView<curandState> states, vec3 dir, float angle, ArrayView<vec3> out,
                                        ArrayView<vec3> out_sphere)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= states.size()) return;
    curandState s   = states[tid];
    out[tid]        = CUDA::sampleCone(dir, angle, s);
    out_sphere[tid] = CUDA::sphericalRand(2.0f, s);
}

// Regression test: sampleCone did not compile in device code (it called the host function Saiga::rotation). It also
// rotated the samples of sampleUnitCone, which are around the y axis, as if they were around the z axis.
TEST(CudaRandom, SampleCone)
{
    const int n = 2000;
    thrust::device_vector<curandState> states(n);
    CUDA::initRandom(states, 123);

    for (vec3 dir : {vec3(0, 0, 1), vec3(1, 0, 0), vec3(0, 1, 0), vec3(1, 2, -3).normalized()})
    {
        float angle = 0.2f;
        thrust::device_vector<vec3> d_out(n), d_sphere(n);
        SampleConeKernel<<<iDivUp(n, 128), 128>>>(states, dir, angle, d_out, d_sphere);
        thrust::host_vector<vec3> h_out    = d_out;
        thrust::host_vector<vec3> h_sphere = d_sphere;
        int outside                        = 0;
        for (int i = 0; i < n; ++i)
        {
            vec3 v = h_out[i];
            EXPECT_NEAR(v.norm(), 1, 1e-4);
            if (v.dot(dir) < std::cos(angle) - 1e-4) outside++;
            EXPECT_NEAR(h_sphere[i].norm(), 2, 1e-4);
        }
        EXPECT_EQ(outside, 0) << dir.transpose();
    }
}

// ======================== Small matrix inverse ========================

TEST(CudaMath, InverseSymmetric)
{
    const int N = 5;
    // A = B * B^T + I is symmetric positive definite.
    double B[N][N];
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) B[i][j] = Random::sampleDouble(-1, 1);
    double A[N * N];
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
        {
            double s = (i == j) ? 1 : 0;
            for (int k = 0; k < N; ++k) s += B[i][k] * B[j][k];
            A[i * N + j] = s;
        }

    auto check_inverse = [&](const double* Ainv)
    {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
            {
                double s = 0;
                for (int k = 0; k < N; ++k) s += A[i * N + k] * Ainv[k * N + j];
                EXPECT_NEAR(s, i == j ? 1 : 0, 1e-10);
            }
    };

    double full[N * N];
    CUDA::invertSymmetric<double, N, CUDA::Full>(A, full);
    check_inverse(full);

    double inplace[N * N];
    std::copy(A, A + N * N, inplace);
    CUDA::invertSymmetric<double, N, CUDA::Full>(inplace);
    check_inverse(inplace);

    double half[N * (N + 1) / 2];
    for (int i = 0, k = 0; i < N; ++i)
        for (int j = 0; j <= i; ++j) half[k++] = A[i * N + j];
    CUDA::invertSymmetric<double, N, CUDA::Half>(half);
    double half_full[N * N];
    for (int i = 0; i < N; ++i)
        for (int j = 0; j <= i; ++j) half_full[i * N + j] = half_full[j * N + i] = half[i * (i + 1) / 2 + j];
    check_inverse(half_full);
}

TEST(CudaMath, Inverse3x3)
{
    double A[9] = {4, 1, 2, 1, 5, 3, 2, 3, 6};
    double Ainv[9];
    CUDA::inverse3x3(A, Ainv);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
        {
            double s = 0;
            for (int k = 0; k < 3; ++k) s += A[i * 3 + k] * Ainv[k * 3 + j];
            EXPECT_NEAR(s, i == j ? 1 : 0, 1e-12);
        }

    double half[6] = {4, 1, 5, 2, 3, 6};
    double half_inv[6];
    CUDA::inverse3x3Symmetric(half, half_inv);
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j <= i; ++j) EXPECT_NEAR(half_inv[i * (i + 1) / 2 + j], Ainv[i * 3 + j], 1e-12);
}

// ======================== Memory, streams and events ========================

TEST(CudaMemory, PinnedVector)
{
    const int n = 100000;
    pinned_vector<int> h(n);
    for (int i = 0; i < n; ++i) h[i] = i * 3 - 7;

    cudaPointerAttributes attributes;
    CHECK_CUDA_ERROR(cudaPointerGetAttributes(&attributes, h.data().get()));
    EXPECT_EQ(attributes.type, cudaMemoryTypeHost);

    thrust::device_vector<int> d = h;
    pinned_vector<int> h2(n, 0);
    CUDA::CudaStream stream;
    CHECK_CUDA_ERROR(cudaMemcpyAsync(h2.data().get(), d.data().get(), n * sizeof(int), cudaMemcpyDeviceToHost, stream));
    stream.synchronize();
    for (int i = 0; i < n; ++i) ASSERT_EQ(h2[i], h[i]);
}

TEST(CudaMemory, TmpAllocator)
{
    Saiga::memory_resource resource;
    thrust::mr::allocator<char, Saiga::memory_resource> alloc(&resource);

    for (int n : {1000, 100000, 50})
    {
        auto h_data = RandomVector<int>(n, -100000, 100000);
        thrust::device_vector<int> d_data(h_data);
        thrust::sort(thrust::cuda::par(alloc), d_data.begin(), d_data.end());
        std::sort(h_data.begin(), h_data.end());
        EXPECT_EQ(ToHost(d_data), h_data);
    }
}

__global__ static void SpinAndWrite(int* data, int value, long long cycles)
{
    long long start = clock64();
    while (clock64() - start < cycles)
    {
    }
    *data = value;
}

__global__ static void ReadPlusOne(const int* in, int* out)
{
    *out = *in + 1;
}

TEST(CudaStream, EventOrdering)
{
    CUDA::CudaStream s1, s2;
    thrust::device_vector<int> d(2, 0);
    int* ptr = d.data().get();

    CUDA::CudaEvent start, written;
    start.record(s1);
    SpinAndWrite<<<1, 1, 0, s1>>>(ptr, 41, 10 * 1000 * 1000);
    written.record(s1);
    s2.waitForEvent(written);
    ReadPlusOne<<<1, 1, 0, s2>>>(ptr, ptr + 1);
    s2.synchronize();

    EXPECT_TRUE(written.isCompleted());
    EXPECT_EQ(int(d[1]), 42);
    float ms = CUDA::CudaEvent::elapsedTime(start, written);
    EXPECT_GT(ms, 0);

    // A copied event is a new event.
    CUDA::CudaEvent copy(written);
    EXPECT_NE(copy.event, written.event);
}

TEST(CudaStream, Timers)
{
    thrust::device_vector<int> d(1);
    float time = -1;
    {
        CUDA::ScopedTimer timer(time);
        SpinAndWrite<<<1, 1>>>(d.data().get(), 0, 1000);
    }
    EXPECT_GE(time, 0);

    CUDA::CudaStream stream;
    CUDA::MultiFrameTimer timer(stream);
    int valid = 0;
    for (int i = 0; i < 5; ++i)
    {
        timer.startTimer();
        SpinAndWrite<<<1, 1, 0, stream>>>(d.data().get(), 1, 100000);
        float t = timer.stopTimer(true);
        if (t >= 0) valid++;
    }
    EXPECT_GE(valid, 3);
}

}  // namespace Saiga

int main(int argc, char** argv)
{
    Saiga::CUDA::initCUDA();
    Saiga::Random::setSeed(9283);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
