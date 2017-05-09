#include "saiga/cuda/cudaHelper.h"

#include "saiga/util/assert.h"

using std::cout;
using std::endl;

namespace CUDA {


void assert_no_cuda_error(cudaError_t code,const char *file, int line) {
    if (code != cudaSuccess) {
        std::cout<<"GPUassert: "<< cudaGetErrorString(code) << " " << file << " " << line << std::endl;
        SAIGA_ASSERT(0);
    }
}


// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
        { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
        { 0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
        {   -1, -1 }
    };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one to run properly
    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index-1].Cores);
    return nGpuArchCoresPerSM[index-1].Cores;
}
//copied from helper_cuda.h in the samples
// This function returns the best GPU (with maximum GFLOPS)
inline int gpuGetMaxGflopsDeviceId()
{
    int current_device     = 0, sm_per_multiproc  = 0;
    int max_perf_device    = 0;
    int device_count       = 0, best_SM_arch      = 0;
    int devices_prohibited = 0;

    unsigned long long max_compute_perf = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceCount(&device_count);

    CHECK_CUDA_ERROR(cudaGetDeviceCount(&device_count));

    if (device_count == 0)
    {
        fprintf(stderr, "gpuGetMaxGflopsDeviceId() CUDA error: no devices supporting CUDA.\n");
        exit(EXIT_FAILURE);
    }

    // Find the best major SM Architecture GPU device
    while (current_device < device_count)
    {
        cudaGetDeviceProperties(&deviceProp, current_device);

        // If this GPU is not running on Compute Mode prohibited, then we can add it to the list
        if (deviceProp.computeMode != cudaComputeModeProhibited)
        {
            if (deviceProp.major > 0 && deviceProp.major < 9999)
            {
                best_SM_arch = std::max(best_SM_arch, deviceProp.major);
            }
        }
        else
        {
            devices_prohibited++;
        }

        current_device++;
    }

    if (devices_prohibited == device_count)
    {
        fprintf(stderr, "gpuGetMaxGflopsDeviceId() CUDA error: all devices have compute mode prohibited.\n");
        exit(EXIT_FAILURE);
    }

    // Find the best CUDA capable GPU device
    current_device = 0;

    while (current_device < device_count)
    {
        cudaGetDeviceProperties(&deviceProp, current_device);

        // If this GPU is not running on Compute Mode prohibited, then we can add it to the list
        if (deviceProp.computeMode != cudaComputeModeProhibited)
        {
            if (deviceProp.major == 9999 && deviceProp.minor == 9999)
            {
                sm_per_multiproc = 1;
            }
            else
            {
                sm_per_multiproc = _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor);
            }

            unsigned long long compute_perf  = (unsigned long long) deviceProp.multiProcessorCount * sm_per_multiproc * deviceProp.clockRate;

            if (compute_perf  > max_compute_perf)
            {
                // If we find GPU with SM major > 2, search only these
                if (best_SM_arch > 2)
                {
                    // If our device==dest_SM_arch, choose this, or else pass
                    if (deviceProp.major == best_SM_arch)
                    {
                        max_compute_perf  = compute_perf;
                        max_perf_device   = current_device;
                    }
                }
                else
                {
                    max_compute_perf  = compute_perf;
                    max_perf_device   = current_device;
                }
            }
        }

        ++current_device;
    }

    return max_perf_device;
}


void initCUDA(){

    int runtimeVersion;
    cudaRuntimeGetVersion(&runtimeVersion);
    int driverVersion;
    cudaDriverGetVersion(&driverVersion);

    cout << "CUDA Runtime Version: " << runtimeVersion << endl;
    cout << "CUDA Driver Version: " << driverVersion << endl;

    int deviceCount;
    CHECK_CUDA_ERROR( cudaGetDeviceCount(&deviceCount));
    SAIGA_ASSERT(deviceCount > 0);


    /* This will pick the best possible CUDA capable device */
    int devID;
    devID = gpuGetMaxGflopsDeviceId();
    CHECK_CUDA_ERROR(cudaSetDevice(devID));


    cudaDeviceProp deviceProp;
    CHECK_CUDA_ERROR(cudaGetDeviceProperties(&deviceProp, devID));
    /* Statistics about the GPU device */
    cout << "Device Number: " << devID << endl;
    cout << "  Device name: " << deviceProp.name << endl;
    cout << "  Compute capabilities: " << deviceProp.major << "." << deviceProp.minor << endl;
    cout << "  Global Memory: " << deviceProp.totalGlobalMem << endl;
    cout << "  Shared Memory: " << deviceProp.sharedMemPerBlock << endl;
    cout << "  Constant Memory: " << deviceProp.totalConstMem << endl;
    cout << "  Warp Size: " << deviceProp.warpSize << endl;
    cout << "  Max Threads per Block: " << deviceProp.maxThreadsPerBlock << endl;
    cout << "  Max Threads per Multi Processor: " << deviceProp.maxThreadsPerMultiProcessor << endl;
    cout << "  32-Bit Registers per Block: " << deviceProp.regsPerBlock << endl;
    cout << "  Multi-Processors: " << deviceProp.multiProcessorCount << endl;
    cout << "  Memory Clock Rate (KHz): " << deviceProp.memoryClockRate << endl;
    cout << "  Memory Bus Width (bits): " << deviceProp.memoryBusWidth << endl;


    //In this calculation, we convert the memory clock rate to Hz,
    //multiply it by the interface width (divided by 8, to convert bits to bytes)
    //and multiply by 2 due to the double data rate. Finally, we divide by 109 to convert the result to GB/s.
    double clockRateHz = deviceProp.memoryClockRate * 1000.0;
    cout << "  Theoretical Memory Bandwidth (GB/s): " << 2.0*clockRateHz*(deviceProp.memoryBusWidth/8)/1.0e9 << endl;
    cout << endl;
}

void destroyCUDA(){
    // cudaDeviceReset causes the driver to clean up all state. While
    // not mandatory in normal operation, it is good practice.  It is also
    // needed to ensure correct operation when the application is being
    // profiled. Calling cudaDeviceReset causes all profile data to be
    // flushed before the application exits
    cudaDeviceReset();
}

extern void testCuda();
extern void testThrust();

void runTests(){
    testCuda();
    testThrust();
}
}
