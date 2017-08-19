/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/cuda/tests/test.h"
#include "saiga/cuda/tests/test_helper.h"
#include "saiga/cuda/thread_info.h"
#include "saiga/cuda/cudaHelper.h"
#include "saiga/time/timer.h"
#include "saiga/image/image.h"
#include <algorithm>

namespace Saiga {
namespace CUDA {


//nvcc $CPPFLAGS -ptx -src-in-ptx -gencode=arch=compute_52,code=compute_52 -g -std=c++11 --expt-relaxed-constexpr integrate_test.cu


void imageProcessingTest(){
    CUDA_SYNC_CHECK_ERROR();
    size_t N = 100 * 1000 * 1000;
    size_t readWrites = N * 2 * sizeof(int);

    CUDA::PerformanceTestHelper pth("Memcpy", readWrites);

    thrust::device_vector<int> src(N);
    thrust::device_vector<int> dest(N);


    Image img;


    CUDA_SYNC_CHECK_ERROR();

}

}
}
