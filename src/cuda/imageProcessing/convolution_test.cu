/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/cuda/imageProcessing/convolution.h"
#include "saiga/cuda/tests/test_helper.h"
#include "saiga/cuda/tests/test.h"
#include "saiga/cuda/thread_info.h"
#include "saiga/cuda/cudaHelper.h"
#include "saiga/time/timer.h"

using std::cout;
using std::endl;

namespace Saiga {
namespace CUDA {

__constant__ float d_Kernel[MAX_RADIUS*2+1];

#define KERNEL_RADIUS 3

////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
#define   ROWS_BLOCKDIM_X 128
#define   ROWS_BLOCKDIM_Y 1
#define ROWS_RESULT_STEPS 8
#define   ROWS_HALO_STEPS 1

__global__ void convolutionRowsKernel2(
        float *d_Dst,
        float *d_Src,
        int imageW,
        int imageH,
        int pitch
        )
{
    __shared__ float s_Data[ROWS_BLOCKDIM_Y][(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

    //Offset to the left halo edge
    const int baseX = (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X + threadIdx.x;
    const int baseY = blockIdx.y * ROWS_BLOCKDIM_Y + threadIdx.y;

    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;

    //Load main data
#pragma unroll

    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
    {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = d_Src[i * ROWS_BLOCKDIM_X];
    }

    //Load left halo
#pragma unroll

    for (int i = 0; i < ROWS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (baseX >= -i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
    }

    //Load right halo
#pragma unroll

    for (int i = ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS + ROWS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X] = (imageW - baseX > i * ROWS_BLOCKDIM_X) ? d_Src[i * ROWS_BLOCKDIM_X] : 0;
    }

    //Compute and store results
    __syncthreads();
#pragma unroll

    for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; i++)
    {
        float sum = 0;

#pragma unroll

        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
        {
            sum += d_Kernel[KERNEL_RADIUS + j] * s_Data[threadIdx.y][threadIdx.x + i * ROWS_BLOCKDIM_X + j];
        }

        d_Dst[i * ROWS_BLOCKDIM_X] = sum;
    }
}


template<typename T, int RADIUS, int BLOCK_W, int BLOCK_H>
__global__ void convolutionRowsKernel(
        float *d_Dst,
        float *d_Src,
        int imageW,
        int imageH,
        int pitch
        )
{
    using vector_type = int2;
    const int elements_per_vector = sizeof(vector_type) / sizeof(T);

    const int shared_block_width = BLOCK_W + ( KERNEL_RADIUS * 2 / elements_per_vector);

    //estimated shared memory per block = 1 * (128+ (4*2/2)) * sizeof(int2) = 512 * sizeof(int2) = 2048
    //per mp: 2048 * (2048/128) = 2048 * 16 = 32768 (=100% occupancy)
    __shared__ vector_type s_Data[BLOCK_H*shared_block_width];

    vector_type* src2 = reinterpret_cast<vector_type*>(d_Src);
    vector_type* dest2 = reinterpret_cast<vector_type*>(d_Dst);

    T* s_Data2 = reinterpret_cast<T*>(s_Data);

    int imageWV = imageW / elements_per_vector;
    int pitchV = pitch / elements_per_vector;


    const int tx = threadIdx.x;
    const int ty = threadIdx.y;

    const int xp = blockIdx.x*BLOCK_W + tx;
    const int yp = blockIdx.y*BLOCK_H + ty;

    //    const int baseX = blockIdx.x*BLOCK_W - KERNEL_RADIUS;

    const int baseXV = blockIdx.x * BLOCK_W - KERNEL_RADIUS / elements_per_vector;


    for (int i = threadIdx.x; i < BLOCK_W + (KERNEL_RADIUS * 2 / elements_per_vector); i+=BLOCK_W)
    {
        int x = baseXV + i;
        x = min(max(0,x),imageWV-1);

        auto v = src2[yp * pitchV + x];

        if(baseXV + i < 0){
            v.y = v.x;
        }
        if(baseXV + i >= imageWV){
            v.x = v.y;
        }

        s_Data[threadIdx.y*shared_block_width+i] = v;
    }


    //Compute and store results
    __syncthreads();

    T sum[2];
    sum[0] = 0;
    sum[1] = 0;

    for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
    {
        auto xoffset = threadIdx.x * elements_per_vector;
        auto yoffset = threadIdx.y * shared_block_width * elements_per_vector;

        sum[0] += d_Kernel[KERNEL_RADIUS + j] * s_Data2[yoffset + xoffset + (KERNEL_RADIUS + j)];
        sum[1] += d_Kernel[KERNEL_RADIUS + j] * s_Data2[yoffset + xoffset + (KERNEL_RADIUS + j) + 1];
    }

    dest2[yp * pitchV + xp] = *reinterpret_cast<vector_type*>(sum);

}
extern "C" void convolutionRowsGPU(
        float *d_Dst,
        float *d_Src,
        int imageW,
        int imageH
        )
{
    assert(ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= KERNEL_RADIUS);
    assert(imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0);
    assert(imageH % ROWS_BLOCKDIM_Y == 0);

    dim3 blocks(imageW / (ROWS_BLOCKDIM_X) / 2, imageH / ROWS_BLOCKDIM_Y);
    //    dim3 blocks(16,1);
    dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);

    convolutionRowsKernel<float,KERNEL_RADIUS,ROWS_BLOCKDIM_X,ROWS_BLOCKDIM_Y>
            <<<blocks, threads>>>(
                                    d_Dst,
                                    d_Src,
                                    imageW,
                                    imageH,
                                    imageW
                                    );
    CUDA_SYNC_CHECK_ERROR();
}



////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
#define   COLUMNS_BLOCKDIM_X 4
#define   COLUMNS_BLOCKDIM_Y 64
#define COLUMNS_RESULT_STEPS 8
#define   COLUMNS_HALO_STEPS 1

__global__ void convolutionColumnsKernel2(
        float *d_Dst,
        float *d_Src,
        int imageW,
        int imageH,
        int pitch
        )
{
    __shared__ float s_Data[COLUMNS_BLOCKDIM_X][(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

    //Offset to the upper halo edge
    const int baseX = blockIdx.x * COLUMNS_BLOCKDIM_X + threadIdx.x;
    const int baseY = (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + threadIdx.y;
    d_Src += baseY * pitch + baseX;
    d_Dst += baseY * pitch + baseX;

    //Main data
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = d_Src[i * COLUMNS_BLOCKDIM_Y * pitch];
    }

    //Upper halo
#pragma unroll

    for (int i = 0; i < COLUMNS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y] = (baseY >= -i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
    }

    //Lower halo
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS + COLUMNS_HALO_STEPS; i++)
    {
        s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y]= (imageH - baseY > i * COLUMNS_BLOCKDIM_Y) ? d_Src[i * COLUMNS_BLOCKDIM_Y * pitch] : 0;
    }

    //Compute and store results
    __syncthreads();
#pragma unroll

    for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
    {
        float sum = 0;
#pragma unroll

        for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
        {
            sum += d_Kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.x][threadIdx.y + i * COLUMNS_BLOCKDIM_Y + j];
        }

        d_Dst[i * COLUMNS_BLOCKDIM_Y * pitch] = sum;
    }
}


template<typename T, int RADIUS, int BLOCK_W, int BLOCK_H>
__global__ void convolutionColumnsKernel(
        float *d_Dst,
        float *d_Src,
        int imageW,
        int imageH,
        int pitch
        )
{

    using vector_type = int2;
    const int elements_per_vector = sizeof(vector_type) / sizeof(T);

    const int shared_block_width = BLOCK_H + ( KERNEL_RADIUS * 2 );

    //estimated shared memory per block = 1 * (128+ (4*2/2)) * sizeof(int2) = 512 * sizeof(int2) = 2048
    //per mp: 2048 * (2048/128) = 2048 * 16 = 32768 (=100% occupancy)
    __shared__ vector_type s_Data[BLOCK_W*shared_block_width];

    vector_type* src2 = reinterpret_cast<vector_type*>(d_Src);
    vector_type* dest2 = reinterpret_cast<vector_type*>(d_Dst);

    //    T* s_Data2 = reinterpret_cast<T*>(s_Data);

    //    int imageWV = imageW / elements_per_vector;
    int pitchV = pitch / elements_per_vector;


    //    const int tx = threadIdx.x;
    //    const int ty = threadIdx.y;

    const int tx = threadIdx.y;
    const int ty = threadIdx.x;

    const int xp = blockIdx.x*BLOCK_W + tx;
    const int yp = blockIdx.y*BLOCK_H + ty;

    //    const int baseX = blockIdx.x*BLOCK_W - KERNEL_RADIUS;

    //    const int baseXV = blockIdx.x * BLOCK_W - KERNEL_RADIUS / elements_per_vector;



    const int baseY = blockIdx.y*BLOCK_H - KERNEL_RADIUS;


    for (int i = ty; i < BLOCK_H + (KERNEL_RADIUS * 2); i+=BLOCK_H)
    {
        int y = baseY + i;
        y = min(max(0,y),imageH-1);
        s_Data[tx*shared_block_width + i] = src2[y * pitchV + xp];

        //        vector_type v;
        //        v.x = 1;
        //        v.y = 1;
        //        s_Data[tx*shared_block_width + i] = v;

        //                if(blockIdx.x == 0 && blockIdx.y == 0){
        //                    printf("%d %d %d %d \n",threadIdx.x,threadIdx.y,y * pitchV + xp,threadIdx.x*shared_block_width + i);
        //                }
    }


    //Compute and store results
    __syncthreads();



    T sum[2];
    sum[0] = 0;
    sum[1] = 0;

#pragma unroll
    for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
        //    for (int j = -0; j <= 0; j++)
    {
        int i = j + KERNEL_RADIUS;
        //        i = (i + threadIdx.x) % (KERNEL_RADIUS*2+1);

        auto xoffset = tx * shared_block_width;
        auto yoffset = ty;
        auto v = s_Data[yoffset + xoffset + i];
        //        auto v = s_Data2[ (yoffset + xoffset + i) * elements_per_vector];
        //        auto v2 = s_Data2[ (yoffset + xoffset + i) * elements_per_vector + 1];

        //        auto v = s_Data2[0];
        //        auto v2 = s_Data2[0];

        float k = d_Kernel[i];
        //        float k = i;
        sum[0]  += k * reinterpret_cast<T*>(&v)[0];
        sum[1]  += k * reinterpret_cast<T*>(&v)[1];

        //        sum[0] += v;
        //        sum[1] += v2;
    }

    dest2[yp * pitchV + xp] = *reinterpret_cast<vector_type*>(sum);

}
extern "C" void convolutionColumnsGPU(
        float *d_Dst,
        float *d_Src,
        int imageW,
        int imageH
        )
{
    assert(COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= KERNEL_RADIUS);
    assert(imageW % COLUMNS_BLOCKDIM_X == 0);
    assert(imageH % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0);

    dim3 blocks(imageW / COLUMNS_BLOCKDIM_X / 2, imageH / (COLUMNS_BLOCKDIM_Y));
    //    dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);
    dim3 threads(COLUMNS_BLOCKDIM_Y, COLUMNS_BLOCKDIM_X);

    convolutionColumnsKernel<float,KERNEL_RADIUS,COLUMNS_BLOCKDIM_X,COLUMNS_BLOCKDIM_Y>
            <<<blocks, threads>>>(
                                    d_Dst,
                                    d_Src,
                                    imageW,
                                    imageH,
                                    imageW
                                    );
    CUDA_SYNC_CHECK_ERROR();
}



template<typename T, int RADIUS, int BLOCK_W, int BLOCK_H>
__global__ static
void singlePassConvolve(ImageView<T> src, ImageView<T> dst)
{
    __shared__ float buffer[(BLOCK_W + 2*RADIUS)*BLOCK_H];
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int xp = blockIdx.x*BLOCK_W + tx;
    const int yp = blockIdx.y*BLOCK_H + ty;

    float *kernel = d_Kernel;

    int nx = min(max(0,xp-RADIUS),src.width-1);

    float *buff = buffer + ty*(BLOCK_W + 2*RADIUS);
    //    int h = src.height-1;
    //    int pitch = src.pitch;

    if (yp<src.height){
        float sum = 0;
        for (int j=-RADIUS;j<=RADIUS;j++){
            int ny = min(max(0,yp+j),src.height-1);
            sum += src(nx,ny) * kernel[j+RADIUS];
        }
        buff[tx] = sum;
    }

    __syncthreads();

    if (tx<BLOCK_W && xp<src.width && yp<src.height) {
        float sum = 0;
        for (int j=-RADIUS;j<=RADIUS;j++){
            int id = tx + j + RADIUS;
            sum += buff[id] * kernel[j+RADIUS];
        }
        dst(xp,yp) = sum;
    }
}



void convolutionTest(){
    CUDA_SYNC_CHECK_ERROR();


    const int kernel_radius = KERNEL_RADIUS;
    const int kernel_size = kernel_radius * 2 + 1;
    float sigma = 2.0f;
//    int h = 256;
    int h = 2048;
    int w = h * 2;

    size_t N = w * h;
    size_t readWrites = N * 2 * sizeof(float);

    Saiga::CUDA::PerformanceTestHelper pth("convolutionTest radius: " + std::to_string(kernel_radius), readWrites);

    thrust::device_vector<float> src(N,0.1);
    thrust::device_vector<float> dest(N,0.1);
    thrust::device_vector<float> tmp(N,0.1);

    thrust::host_vector<float> h_src = src;
    thrust::host_vector<float> h_dest = dest;
    thrust::host_vector<float> h_tmp = dest;
    thrust::host_vector<float> h_ref = dest;

    ImageView<float> imgSrc(w,h,thrust::raw_pointer_cast(src.data()));
    ImageView<float> imgDst(w,h,thrust::raw_pointer_cast(dest.data()));
    ImageView<float> imgTmp(w,h,thrust::raw_pointer_cast(tmp.data()));


    ImageView<float> h_imgSrc(w,h,thrust::raw_pointer_cast(h_src.data()));
    ImageView<float> h_imgDst(w,h,thrust::raw_pointer_cast(h_dest.data()));
    ImageView<float> h_imgTmp(w,h,thrust::raw_pointer_cast(h_tmp.data()));

    thrust::host_vector<float> h_kernel(kernel_size);
    float kernelSum = 0.0f;
    float ivar2 = 1.0f/(2.0f*sigma*sigma);
    for (int j=-kernel_radius;j<=kernel_radius;j++) {
        h_kernel[j+kernel_radius] = (float)expf(-(double)j*j*ivar2);
        kernelSum += h_kernel[j+kernel_radius];
    }
    for (int j=-kernel_radius;j<=kernel_radius;j++){
        h_kernel[j+kernel_radius] /= kernelSum;
        //        cout << h_kernel[j+kernel_radius] << endl;
    }

    {
        for(int y = 0; y < h; ++y){
            for(int x = 0; x < w; ++x){
                h_imgSrc(x,y) = (rand()%3) - 1;
            }
        }
        src = h_src;
    }

    //    copyConvolutionKernel(h_kernel);
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(d_Kernel, h_kernel.data(), h_kernel.size()*sizeof(float)));

    {
        float time;
        {
            Saiga::ScopedTimer<float> t(&time);
#pragma omp parallel for
            for(int y = 0; y < h; ++y){
                for(int x = 0; x < w; ++x){
                    float sum = 0;
                    for (int j=-kernel_radius;j<=kernel_radius;j++){
                        int ny = std::min(std::max(0,y+j),h-1);
                        float innerSum = 0;
                        for (int i=-kernel_radius;i<=kernel_radius;i++){
                            int nx = std::min(std::max(0,x+i),w-1);
                            innerSum += h_imgSrc(nx,ny) * h_kernel[i+kernel_radius];
                        }
                        sum += innerSum * h_kernel[j+kernel_radius];
                    }
                    h_imgDst(x,y) = sum;
                }
            }


        }
        pth.addMeassurement("CPU Convolve",time);
        h_ref = h_dest;
        //        cout << "h_ref[0]=" << h_ref[0] << endl;
    }

    {
        float time;
        {
            Saiga::ScopedTimer<float> t(&time);
            #pragma omp parallel for
            for(int y = 0; y < h; ++y){
                for(int x = 0; x < w; ++x){
                    float sum = 0;
                    for (int j=-kernel_radius;j<=kernel_radius;j++){
                        int nx = std::min(std::max(0,x+j),w-1);
                        sum += h_imgSrc(nx,y) * h_kernel[j+kernel_radius];
                    }
                    h_imgTmp(x,y) = sum;
                }
            }

#pragma omp parallel for
            for(int x = 0; x < w; ++x){
                for(int y = 0; y < h; ++y){
                    float sum = 0;
                    for (int j=-kernel_radius;j<=kernel_radius;j++){
                        int ny = std::min(std::max(0,y+j),h-1);
                        sum += h_imgTmp(x,ny) * h_kernel[j+kernel_radius];
                    }
                    h_imgDst(x,y) = sum;
                }
            }
        }
        pth.addMeassurement("CPU Convolve Separate",time);
        SAIGA_ASSERT(h_ref == h_dest);
    }



    {
        dest = src;
        float time;
        {
            Saiga::CUDA::CudaScopedTimer t(time);
            const int LOWPASS_W = 32;
            const int LOWPASS_H = 16;

            dim3 blocks(Saiga::CUDA::getBlockCount(w, LOWPASS_W), Saiga::CUDA::getBlockCount(h, LOWPASS_H));
            dim3 threads(LOWPASS_W+2*kernel_radius, LOWPASS_H);
            singlePassConvolve<float,kernel_radius,LOWPASS_W,LOWPASS_H> <<<blocks, threads>>>(imgSrc,imgDst);
        }
        pth.addMeassurement("GPU Convolve Single Pass",time);
        thrust::host_vector<float> test = dest;
        for(int i = 0; i < test.size();++i){
            if(std::abs(test[i]-h_ref[i]) > 1e-5){
                cout << "error " << i << " " << test[i] << "!=" << h_ref[i] << endl;
                SAIGA_ASSERT(0);
            }
        }
    }

    {
        thrust::device_vector<float> d_kernel = h_kernel;
        dest = src;
        float time;
        {
            Saiga::CUDA::CudaScopedTimer t(time);
            convolveSinglePassSeparate(imgSrc,imgDst,d_kernel,KERNEL_RADIUS);
        }
        pth.addMeassurement("GPU Convolve Single Pass2",time);
        thrust::host_vector<float> test = dest;
        for(int i = 0; i < test.size();++i){
            if(std::abs(test[i]-h_ref[i]) > 1e-5){
                cout << "error " << i << " " << test[i] << "!=" << h_ref[i] << endl;
                SAIGA_ASSERT(0);
            }
        }
    }

    {
        thrust::device_vector<float> d_kernel = h_kernel;
        dest = src;
        float time;
        {
            Saiga::CUDA::CudaScopedTimer t(time);
            convolveSinglePassSeparate2(imgSrc,imgDst,d_kernel,KERNEL_RADIUS);
        }
        pth.addMeassurement("GPU Convolve Single Pass2",time);
        thrust::host_vector<float> test = dest;
        for(int i = 0; i < test.size();++i){
            if(std::abs(test[i]-h_ref[i]) > 1e-5){
                cout << "error " << i << " " << test[i] << "!=" << h_ref[i] << endl;
                SAIGA_ASSERT(0);
            }
        }
    }

    {
        thrust::device_vector<float> d_kernel = h_kernel;
        dest = src;
        float time;
        {
            Saiga::CUDA::CudaScopedTimer t(time);
            convolveSinglePassSeparate3(imgSrc,imgDst,d_kernel,KERNEL_RADIUS);
        }
        pth.addMeassurement("GPU Convolve Single Pass3",time);
        thrust::host_vector<float> test = dest;
        for(int i = 0; i < test.size();++i){
            if(std::abs(test[i]-h_ref[i]) > 1e-5){
                cout << "error " << i << " " << test[i] << "!=" << h_ref[i] << endl;
                SAIGA_ASSERT(0);
            }
        }
    }

    {
        thrust::device_vector<float> d_kernel = h_kernel;
        dest = src;
        float time;
        {
            Saiga::CUDA::CudaScopedTimer t(time);
            convolveSinglePassSeparate4(imgSrc,imgDst,d_kernel,KERNEL_RADIUS);
        }
        pth.addMeassurement("GPU Convolve Single Pass4",time);
        thrust::host_vector<float> test = dest;
        for(int i = 0; i < test.size();++i){
            if(std::abs(test[i]-h_ref[i]) > 1e-5){
                cout << "error " << i << " " << test[i] << "!=" << h_ref[i] << endl;
                SAIGA_ASSERT(0);
            }
        }
    }

    {
        dest = src;
        tmp = src;
        float time1;
        {
            Saiga::CUDA::CudaScopedTimer t(time1);
            convolutionRowsGPU((float*)imgTmp.data,(float*)imgSrc.data,w,h);
        }
        pth.addMeassurement("GPU Convolve Separate Row",time1);
        float time2;
        {
            Saiga::CUDA::CudaScopedTimer t(time2);
            convolutionColumnsGPU((float*)imgDst.data,(float*)imgTmp.data,w,h);
        }
        pth.addMeassurement("GPU Convolve Separate Col",time2);
        pth.addMeassurement("GPU Convolve Separate Total",time1+time2);



        thrust::host_vector<float> test = dest;
        for(int i = 0; i < test.size();++i){
            if(std::abs(test[i]-h_ref[i]) > 1e-5){
                cout << "error " << i << " " << test[i] << "!=" << h_ref[i] << " " << h_tmp[i] << endl;
                SAIGA_ASSERT(0);
            }
        }
    }

    {
        float time;
        {
            Saiga::CUDA::CudaScopedTimer t(time);
            cudaMemcpy(thrust::raw_pointer_cast(dest.data()),thrust::raw_pointer_cast(src.data()),N * sizeof(int),cudaMemcpyDeviceToDevice);

        }
        pth.addMeassurement("cudaMemcpy", time);
    }
    CUDA_SYNC_CHECK_ERROR();

}

}
}
