#include "saiga/cuda/math/inverse.h"
#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/tests/test.h"
#include "saiga/cuda/tests/test_helper.h"
#include "saiga/cuda/thread_info.h"
#include "saiga/core/time/timer.h"
#include "saiga/core/util/assert.h"


#if defined(SAIGA_USE_EIGEN) && defined(SAIGA_EIGEN_AND_CUDA)


#    include "saiga/extra/eigen/eigen.h"

using matrix_t = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

namespace Saiga
{
namespace CUDA
{
template <typename T, int N, unsigned int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE) __global__ void invertMatrices(Saiga::ArrayView<T> data, Saiga::ArrayView<T> result)
{
    const int matrixElements = N * N;

    Saiga::CUDA::ThreadInfo<BLOCK_SIZE> ti;
    // grid stride loop
    for (auto id = ti.thread_id * matrixElements; id < data.size(); id += ti.grid_size * matrixElements)
    {
        T l[matrixElements];
        for (int i = 0; i < matrixElements; ++i)
        {
            l[i] = data[id + i];
        }

        invertSymmetric<T, N, Full>(l);
        //        choleskyKernel2<T,N>(l);
        //        inverseTriangularKernel<T,N,Full>(l);
        //        multLTL<T,N,Full>(l);

        for (int i = 0; i < matrixElements; ++i)
        {
            result[id + i] = l[i];
        }
    }
}

template <typename T, int N, unsigned int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE) __global__ void invertMatrices2(Saiga::ArrayView<T> data, Saiga::ArrayView<T> result)
{
    const int matrixElements  = N * N;
    const int elementsPerWarp = matrixElements * WARP_SIZE;
    //    const int elementsPerBlock = matrixElements * BLOCK_SIZE;

    __shared__ double buffer[BLOCK_SIZE][matrixElements];

    Saiga::CUDA::ThreadInfo<BLOCK_SIZE> ti;
    // grid stride loop
    for (auto id = ti.thread_id * matrixElements; id < data.size(); id += ti.grid_size * matrixElements)
    {
        //        auto matrixId = ti.thread_id;
        //        auto globalOffset = matrixId * matrixElements;
        auto localMatrixId = ti.local_thread_id;  // id in shared buffer

        auto warpStart = ti.warp_id * elementsPerWarp;
        //        auto sharedBlockWarpStart = ti.warp_lane * elementsPerWarp;


        // strided copy
        for (auto e = ti.lane_id; e < elementsPerWarp; e += WARP_SIZE)
        {
            auto localMatrix                 = ti.warp_lane * WARP_SIZE + e / matrixElements;
            auto localOffset                 = e % matrixElements;
            buffer[localMatrix][localOffset] = data[warpStart + e];
        }



        T l[matrixElements];
        for (int i = 0; i < matrixElements; ++i)
        {
            l[i] = buffer[localMatrixId][i];
        }

        invertSymmetric<T, N, Full>(l);


        for (int i = 0; i < matrixElements; ++i)
        {
            buffer[localMatrixId][i] = l[i];
        }

        for (auto e = ti.lane_id; e < elementsPerWarp; e += WARP_SIZE)
        {
            auto localMatrix      = ti.warp_lane * WARP_SIZE + e / matrixElements;
            auto localOffset      = e % matrixElements;
            result[warpStart + e] = buffer[localMatrix][localOffset];
        }
    }
}

template <typename T, int N, unsigned int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE) __global__ void invertMatrices3(Saiga::ArrayView<T> data, Saiga::ArrayView<T> result)
{
    const int matrixElements = N * N;
    //    const int elementsPerWarp = matrixElements * WARP_SIZE;
    //    const int elementsPerBlock = matrixElements * BLOCK_SIZE;

    //    __shared__ double buffer[BLOCK_SIZE][matrixElements];



    //    const int elementsPerWarp = matrixElements * WARP_SIZE;
    const int tiles            = 2;
    const int elementsPerTile  = matrixElements / tiles;
    const int elementsPerBlock = matrixElements * BLOCK_SIZE;

    //    __shared__ double buffer[elementsPerBlock];
    __shared__ T buffer[BLOCK_SIZE][elementsPerTile];

    Saiga::CUDA::ThreadInfo<BLOCK_SIZE> ti;
    // grid stride loop
    for (auto id = ti.thread_id * matrixElements; id < data.size(); id += ti.grid_size * matrixElements)
    {
        T l[matrixElements];

        auto localMatrixId = ti.local_thread_id;  // id in shared buffer
        //        auto warpStart = ti.warp_id * elementsPerWarp;
        auto blockStart = ti.block_id * elementsPerBlock;

        auto warpOffset = ti.warp_lane * WARP_SIZE;  // start matrix of this warp in block local shared memory

        for (int t = 0; t < tiles; ++t)
        {
            auto tileOffset = t * elementsPerTile;
            // strided copy
            for (auto e = ti.lane_id; e < elementsPerTile * WARP_SIZE; e += WARP_SIZE)
            {
                auto localMatrix                 = warpOffset + e / elementsPerTile;
                auto localOffset                 = e % elementsPerTile;
                auto globalIndex                 = blockStart + localMatrix * matrixElements + tileOffset + localOffset;
                buffer[localMatrix][localOffset] = data[globalIndex];
            }

            for (int i = 0; i < elementsPerTile; ++i)
            {
                l[i + tileOffset] = buffer[localMatrixId][i];
            }
        }


        //        choleskyKernel2<T,N>(l);
        invertSymmetric<T, N>(l);


        for (int t = 0; t < tiles; ++t)
        {
            auto tileOffset = t * elementsPerTile;

            for (int i = 0; i < elementsPerTile; ++i)
            {
                buffer[localMatrixId][i] = l[i + tileOffset];
            }
            // strided copy
            for (auto e = ti.lane_id; e < elementsPerTile * WARP_SIZE; e += WARP_SIZE)
            {
                auto localMatrix    = warpOffset + e / elementsPerTile;
                auto localOffset    = e % elementsPerTile;
                auto globalIndex    = blockStart + localMatrix * matrixElements + tileOffset + localOffset;
                result[globalIndex] = buffer[localMatrix][localOffset];
            }
        }
    }
}

template <typename T, int N>
__global__ static void test3(T* buffer)
{
    const int matrixElements = N * N;

    T l[matrixElements];
    for (int i = 0; i < matrixElements; ++i)
    {
        l[i] = buffer[i];
    }

    choleskyKernel2<T, N>(l);

    for (int i = 0; i < matrixElements; ++i)
    {
        buffer[i] = l[i];
    }
}


template <typename T, int N, unsigned int BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE) __global__ void matrixCopy(Saiga::ArrayView<T> data, Saiga::ArrayView<T> result)
{
    //    const int padding = 1;
    const int matrixElements  = N * N;
    const int elementsPerWarp = matrixElements * WARP_SIZE;
    //    const int elementsPerBlock = matrixElements * BLOCK_SIZE;

    //    __shared__ double buffer[elementsPerBlock];
    __shared__ double buffer[BLOCK_SIZE][matrixElements + 0];

    Saiga::CUDA::ThreadInfo<BLOCK_SIZE> ti;
    // grid stride loop
    for (auto id = ti.thread_id * matrixElements; id < data.size(); id += ti.grid_size * matrixElements)
    {
        //        auto matrixId = ti.thread_id;
        //        auto globalOffset = matrixId * matrixElements;
        auto localMatrixId = ti.local_thread_id;  // id in shared buffer

        auto warpStart = ti.warp_id * elementsPerWarp;
        //        auto sharedBlockWarpStart = ti.warp_lane * elementsPerWarp;



#    if 0
//        for(auto e = ti.lane_id; e < elementsPerWarp; e += WARP_SIZE){
//            result[warpStart+e] = data[warpStart+e];
//        }


        //strided copy
        for(auto e = ti.lane_id; e < elementsPerWarp; e += WARP_SIZE){
            auto localMatrix = ti.warp_lane * WARP_SIZE + e / matrixElements;
            auto localOffset = e % matrixElements;
            result[warpStart+e] = buffer[localMatrix][localOffset] = data[warpStart+e];
        }
#    elif 1

#        if 1
        // strided copy
        for (auto e = ti.lane_id; e < elementsPerWarp; e += WARP_SIZE)
        {
            auto localMatrix                 = ti.warp_lane * WARP_SIZE + e / matrixElements;
            auto localOffset                 = e % matrixElements;
            buffer[localMatrix][localOffset] = data[warpStart + e];
        }
#        else
        // linear copy
        for (int i = 0; i < matrixElements; ++i)
        {
            buffer[localMatrixId][i] = data[globalOffset + i];
        }
#        endif

        T l[matrixElements];
        for (int i = 0; i < matrixElements; ++i)
        {
            l[i] = buffer[localMatrixId][i];
        }


        // add something so things don't get optimized away
        for (int i = 0; i < matrixElements; ++i)
        {
            l[i] += 42;
        }

        for (int i = 0; i < matrixElements; ++i)
        {
            buffer[localMatrixId][i] = l[i];
        }

#        if 1
        // strided copy
        for (auto e = ti.lane_id; e < elementsPerWarp; e += WARP_SIZE)
        {
            auto localMatrix      = ti.warp_lane * WARP_SIZE + e / matrixElements;
            auto localOffset      = e % matrixElements;
            result[warpStart + e] = buffer[localMatrix][localOffset];
        }
#        else
        // linear copy
        for (int i = 0; i < matrixElements; ++i)
        {
            result[globalOffset + i] = buffer[localMatrixId][i];
        }
#        endif


#    else

        auto localStart = ti.thread_id * matrixElements;

        T l[matrixElements];
        for (int i = 0; i < matrixElements; ++i)
        {
            l[i] = data[localStart + i];
        }


        // add something so things don't get optimized away
        for (int i = 0; i < matrixElements; ++i)
        {
            l[i] += 42;
        }

        for (int i = 0; i < matrixElements; ++i)
        {
            result[localStart + i] = l[i];
        }


#    endif
    }
}


static void testCorrectness()
{
    cout << "testCorrectness" << endl;
    const int MatrixSize = 3;
    using fixed_matrix_t = Eigen::Matrix<double, MatrixSize, MatrixSize, Eigen::RowMajor>;

    fixed_matrix_t A, Aref, Ares;
    std::vector<double> Ahalf, Ahalfres;

    for (int i = 0; i < MatrixSize; ++i)
    {
        double d = rand() % 10 + 2;
        A(i, i)  = d;
        for (int j = 0; j < i; ++j)
        {
            double d = rand() % 3 - 1;
            A(i, j)  = d;
            A(j, i)  = d;
        }
    }

    for (int i = 0; i < MatrixSize; ++i)
    {
        for (int j = 0; j <= i; ++j)
        {
            Ahalf.push_back(A(i, j));
        }
    }

    Ahalfres.resize(Ahalf.size());
    SAIGA_ASSERT(Ahalf.size() == 6);

    cout << "A: " << endl << A << endl;

    Aref = A.inverse();

    cout << "Eigen inverse: " << endl << Aref << endl;

    invertSymmetric<double, MatrixSize>(A.data(), Ares.data());
    cout << "invertSymmetric: " << endl << Ares << endl;

    invertSymmetric<double, MatrixSize, MatrixIndexOp::Half>(Ahalf.data(), Ahalfres.data());
    cout << "invertSymmetric Half: " << endl;
    for (auto d : Ahalfres)
    {
        cout << d << endl;
    }


    inverse3x3<double>(A.data(), Ares.data());
    cout << "inverse3x3: " << endl << Ares << endl;

    inverse3x3Symmetric<double>(Ahalf.data(), Ahalfres.data());
    cout << "inverse3x3Half: " << endl;
    for (auto d : Ahalfres)
    {
        cout << d << endl;
    }


    cout << endl;
}

// nvcc $CPPFLAGS -I ~/Master/libs/data/include/eigen3/ -ptx -lineinfo -src-in-ptx
// -gencode=arch=compute_52,code=compute_52 -g -std=c++11 --expt-relaxed-constexpr inverse_test.cu nvcc $CPPFLAGS -I
// ~/Master/libs/data/include/eigen3/ -ptx -gencode=arch=compute_52,code=compute_52 -g -std=c++11
// --expt-relaxed-constexpr inverse_test.cu
void inverseTest()
{
    testCorrectness();

    const int MatrixSize  = 4;
    const int MatrixCount = 1000000;

    size_t readWrites = MatrixSize * MatrixSize * MatrixCount * sizeof(double) * 2;
    Saiga::CUDA::PerformanceTestHelper test("Matrix inverse test. MatrixSize: " + std::to_string(MatrixSize) + "x" +
                                                std::to_string(MatrixSize) +
                                                " MatrixCount: " + std::to_string(MatrixCount),
                                            readWrites);
    using fixed_matrix_t = Eigen::Matrix<double, MatrixSize, MatrixSize, Eigen::RowMajor>;
    std::vector<double> data(MatrixSize * MatrixSize * MatrixCount);
    std::vector<double> result = data;



    std::vector<fixed_matrix_t> As(10);
    //    matrix_t A(MatrixSize,MatrixSize);

    for (auto& A : As)
    {
        // create symmetric posivite definite matrix
        for (int i = 0; i < MatrixSize; ++i)
        {
            A(i, i) = rand() % 10 + 2;
            for (int j = 0; j < i; ++j)
            {
                double d = rand() % 3 - 1;
                A(i, j)  = d;
                A(j, i)  = d;
            }
        }
    }


    for (int m = 0; m < MatrixCount; ++m)
    {
        int offset        = m * MatrixSize * MatrixSize;
        fixed_matrix_t& A = As[rand() % As.size()];
        for (int i = 0; i < MatrixSize; ++i)
        {
            for (int j = 0; j < MatrixSize; ++j)
            {
                data[offset + i * MatrixSize + j] = A(i, j);
            }
        }
    }


    thrust::device_vector<double> d_data(data);
    thrust::device_vector<double> d_result(result);


    {
        result = data;
        float time;
        {
            Saiga::ScopedTimer<float> t(&time);
            for (int m = 0; m < MatrixCount; ++m)
            {
                int offset = m * MatrixSize * MatrixSize;
                Eigen::Map<fixed_matrix_t> A(data.data() + offset);
                Eigen::Map<fixed_matrix_t> Ainv(result.data() + offset);
                Ainv = A.inverse();
            }
        }
        test.addMeassurement("Eigen::inverse()", time);
    }

    auto ref = result;


    {
        result = data;
        float time;
        {
            Saiga::ScopedTimer<float> t(&time);
            for (int m = 0; m < MatrixCount; ++m)
            {
                int offset = m * MatrixSize * MatrixSize;
                Eigen::Map<fixed_matrix_t> A(data.data() + offset);
                Eigen::Map<fixed_matrix_t> Ainv(result.data() + offset);
                //                Ainv = A.llt().solve(fixed_matrix_t::Identity());
                Ainv = A.inverse();
            }
        }
        test.addMeassurement("Eigen::LLT::inverse()", time);
    }

    for (int i = 0; i < (int)ref.size(); ++i)
    {
        if (abs(ref[i] - result[i]) > 1e-10)
        {
            SAIGA_ASSERT(0);
        }
    }

    {
        result = data;
        float time;
        {
            Saiga::ScopedTimer<float> t(&time);
            for (int m = 0; m < MatrixCount; ++m)
            {
                int offset = m * MatrixSize * MatrixSize;
                invertSymmetric<double, MatrixSize, Full>(data.data() + offset, result.data() + offset);
            }
        }
        test.addMeassurement("My inverse CPU", time);
    }

    for (int i = 0; i < (int)ref.size(); ++i)
    {
        if (abs(ref[i] - result[i]) > 1e-10)
        {
            SAIGA_ASSERT(0);
        }
    }


    {
        const int BLOCK_SIZE = 128;
        result               = data;
        float time;
        {
            Saiga::CUDA::CudaScopedTimer t(time);
            invertMatrices<double, MatrixSize, BLOCK_SIZE>
                <<<Saiga::CUDA::getBlockCount(MatrixCount, BLOCK_SIZE), BLOCK_SIZE>>>(d_data, d_result);
        }
        test.addMeassurement("My inverse GPU", time);
        CUDA_SYNC_CHECK_ERROR();
    }


    thrust::host_vector<double> test2 = d_result;


    for (int i = 0; i < (int)ref.size(); ++i)
    {
        if (abs(ref[i] - test2[i]) > 1e-10)
        {
            SAIGA_ASSERT(0);
        }
    }



    {
        const int BLOCK_SIZE = 128;
        d_result             = data;
        float time;
        {
            Saiga::CUDA::CudaScopedTimer t(time);
            invertMatrices2<double, MatrixSize, BLOCK_SIZE>
                <<<Saiga::CUDA::getBlockCount(MatrixCount, BLOCK_SIZE), BLOCK_SIZE>>>(d_data, d_result);
        }
        test.addMeassurement("My  inverse2 GPU", time);
        CUDA_SYNC_CHECK_ERROR();
    }


    test2 = d_result;


    for (int i = 0; i < (int)ref.size(); ++i)
    {
        if (abs(ref[i] - test2[i]) > 1e-10)
        {
            cout << ref[i] << " " << test2[i] << endl;
            SAIGA_ASSERT(0);
        }
    }

    {
        const int BLOCK_SIZE = 128;
        d_result             = data;
        float time;
        {
            Saiga::CUDA::CudaScopedTimer t(time);
            invertMatrices3<double, MatrixSize, BLOCK_SIZE>
                <<<Saiga::CUDA::getBlockCount(MatrixCount, BLOCK_SIZE), BLOCK_SIZE>>>(d_data, d_result);
        }
        test.addMeassurement("My  inverse3 GPU", time);
        CUDA_SYNC_CHECK_ERROR();
    }


    test2 = d_result;


    for (int i = 0; i < (int)ref.size(); ++i)
    {
        if (abs(ref[i] - test2[i]) > 1e-10)
        {
            SAIGA_ASSERT(0);
        }
    }

    {
        result = data;
        float time;
        {
            Saiga::CUDA::CudaScopedTimer t(time);
            test3<double, 4><<<1, 1>>>(thrust::raw_pointer_cast(d_data.data()));
        }
        test.addMeassurement("invertMatrices3", time);
        CUDA_SYNC_CHECK_ERROR();
    }

    {
        const int BLOCK_SIZE = 128;
        result               = data;
        float time;
        {
            Saiga::CUDA::CudaScopedTimer t(time);
            matrixCopy<double, MatrixSize, BLOCK_SIZE>
                <<<Saiga::CUDA::getBlockCount(MatrixCount, BLOCK_SIZE), BLOCK_SIZE>>>(d_data, d_result);
        }
        test.addMeassurement("Matrix Copy", time);
        CUDA_SYNC_CHECK_ERROR();
    }


    {
        result = data;
        float time;
        {
            Saiga::CUDA::CudaScopedTimer t(time);
            cudaMemcpy(thrust::raw_pointer_cast(d_result.data()), thrust::raw_pointer_cast(d_data.data()),
                       d_data.size() * sizeof(double), cudaMemcpyDeviceToDevice);
        }
        test.addMeassurement("cudaMemcpy", time);
        CUDA_SYNC_CHECK_ERROR();
    }

    return;
}


}  // namespace CUDA
}  // namespace Saiga

#endif
