#include "saiga/cuda/cudaTimer.h"
#include "saiga/util/assert.h"

using std::cout;
using std::endl;

namespace CUDA {

CudaScopedTimer::CudaScopedTimer(float& time) : time(time){

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
}

CudaScopedTimer::~CudaScopedTimer(){
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}





CudaScopedTimerPrint::CudaScopedTimerPrint(const std::string &name) : name(name){

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
}

CudaScopedTimerPrint::~CudaScopedTimerPrint(){
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float time;
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    std::cout << name << " : " << time << "ms." << std::endl;
}


}
