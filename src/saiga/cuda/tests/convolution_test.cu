/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/cuda/cudaHelper.h"
#include "saiga/cuda/imageProcessing/imageProcessing.h"
#include "saiga/cuda/tests/test.h"
#include "saiga/cuda/tests/test_helper.h"

namespace Saiga
{
namespace CUDA
{
static void checkRes(ImageView<float> ref, ImageView<float> dst)
{
    for (int y = 0; y < ref.rows; ++y)
    {
        for (int x = 0; x < ref.cols; ++x)
        {
#if 0
            cout << dst(y,x) << " ";
#else
            if (std::abs(dst(y, x) - ref(y, x)) > 1e-5)
            {
                std::cout << "error (" << x << "," << y << ") " << dst(y, x) << "!=" << ref(y, x) << std::endl;
                SAIGA_ASSERT(0);
            }
#endif
        }
        //        cout << endl;
    }
}

// static void checkRes2(const thrust::host_vector<float>& ref, const thrust::host_vector<float>& dst){
//    int c = 0;
//    for(int i = 0; i < (int)dst.size();++i){
//        auto refv = 9.0f;
//        if(std::abs(dst[i] - refv) > 1e-5){
//            cout << "error " << (i/2048) << "," << (i%2048) << " " << dst[i] << "!=" << refv << endl;
//            c++;
//            SAIGA_ASSERT(c < 5);
//        }
//    }
//}

template <int KERNEL_RADIUS>
void convolutionTest2(int w, int h)
{
    CUDA_SYNC_CHECK_ERROR();

    size_t N          = w * h;
    size_t readWrites = N * 2 * sizeof(float);


    size_t pitch = Saiga::iAlignUp(sizeof(float) * w, 1024 * 1024);
    size_t size  = pitch * h;


    thrust::device_vector<char> src(size, 0);
    thrust::device_vector<char> dest(size, 0);
    thrust::device_vector<char> tmp(size, 0);

    thrust::host_vector<char> h_src  = src;
    thrust::host_vector<char> h_dest = dest;
    thrust::host_vector<char> h_tmp  = dest;
    thrust::host_vector<char> h_ref  = dest;

    ImageView<float> imgSrc(h, w, pitch, thrust::raw_pointer_cast(src.data()));
    ImageView<float> imgDst(h, w, pitch, thrust::raw_pointer_cast(dest.data()));
    ImageView<float> imgTmp(h, w, pitch, thrust::raw_pointer_cast(tmp.data()));


    ImageView<float> h_imgSrc(h, w, pitch, thrust::raw_pointer_cast(h_src.data()));
    ImageView<float> h_imgDst(h, w, pitch, thrust::raw_pointer_cast(h_dest.data()));
    ImageView<float> h_imgTmp(h, w, pitch, thrust::raw_pointer_cast(h_tmp.data()));

    int its = 50;
    //    float sigma = 2.0f;
    //    thrust::device_vector<float> d_kernel = createGaussianBlurKernel(KERNEL_RADIUS,sigma);
    thrust::device_vector<float> d_kernel(2 * KERNEL_RADIUS + 1, 1.0f);

    thrust::host_vector<float> h_kernel(d_kernel);

    {
        for (int y = 0; y < h; ++y)
        {
            for (int x = 0; x < w; ++x)
            {
                //                h_imgSrc(y,x) = (rand()%3) - 1;
                h_imgSrc(y, x) = 1;
            }
        }
        src = h_src;
    }

#if 0
//    cout << "first pixels: " << h_imgSrc(0,0) << " " << h_imgSrc(0,1) << " " << h_imgSrc(1,0) << " " << h_imgSrc(1,1) << endl;


    int fx = 508 % w;
    int fy = 508 / w;
    cout << "debug pixel: " << fx << " " << fy << endl;
    for (int j=-KERNEL_RADIUS;j<=KERNEL_RADIUS;j++){
                float sum =0;
                            for (int i=-KERNEL_RADIUS;i<=KERNEL_RADIUS;i++){
                                auto v = h_imgSrc.clampedRead(fy+j,fx+i);
                                sum += v;
                               cout << v << " ";
                            }
                            cout << " row sum: " << sum << endl;
    }
#endif

    Saiga::CUDA::PerformanceTestHelper pth("convolutionTest radius=" + std::to_string(KERNEL_RADIUS) +
                                               " ImageSize: " + std::to_string(w) + "x" + std::to_string(h),
                                           readWrites);

    // this takes too long :D
#if 0
    {
        float time;
        {
            Saiga::ScopedTimer<float> t(&time);
            for(int y = 0; y < h; ++y){
                for(int x = 0; x < w; ++x){
                    float sum = 0;
                    for (int j=-KERNEL_RADIUS;j<=KERNEL_RADIUS;j++){
                        float innerSum = 0;
                        for (int i=-KERNEL_RADIUS;i<=KERNEL_RADIUS;i++){
                            innerSum += h_imgSrc.clampedRead(y +j ,x + i) * h_kernel[i+KERNEL_RADIUS];
                        }
                        sum += innerSum * h_kernel[j+KERNEL_RADIUS];
                    }
                    h_imgDst(y,x) = sum;
                }
            }
        }
        pth.addMeassurement("CPU Convolve",time);
        h_ref = h_dest;
    }
#endif

    {
        float time;
        {
            Saiga::ScopedTimer<float> t(&time);
            for (int y = 0; y < h; ++y)
            {
                for (int x = 0; x < w; ++x)
                {
                    float sum = 0;
                    for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
                    {
                        sum += h_imgSrc.clampedRead(y, x + j) * h_kernel[j + KERNEL_RADIUS];
                    }
                    h_imgTmp(y, x) = sum;
                }
            }

            for (int x = 0; x < w; ++x)
            {
                for (int y = 0; y < h; ++y)
                {
                    float sum = 0;
                    for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
                    {
                        sum += h_imgTmp.clampedRead(y + j, x) * h_kernel[j + KERNEL_RADIUS];
                    }
                    h_imgDst(y, x) = sum;
                }
            }
        }
        pth.addMeassurement("CPU Convolve Separate", time);
        h_ref = h_dest;
    }


#if 0
    {
        thrust::device_vector<float> d_kernel = h_kernel;
        dest = src;
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
            convolveSinglePassSeparateOuterLinear(imgSrc,imgDst,d_kernel,KERNEL_RADIUS);
        });


        pth.addMeassurement("convolveSinglePassSeparateOuterLinear",st.median);
        checkRes(h_ref,thrust::host_vector<float>(dest));
    }

    {
        thrust::device_vector<float> d_kernel = h_kernel;
        dest = src;


        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
            convolveSinglePassSeparateOuterHalo(imgSrc,imgDst,d_kernel,KERNEL_RADIUS);
        });
        pth.addMeassurement("convolveSinglePassSeparateOuterHalo",st.median);
        checkRes(h_ref,thrust::host_vector<float>(dest));
    }

    {
        thrust::device_vector<float> d_kernel = h_kernel;
        dest = src;
        auto st = Saiga::measureObject<Saiga::CUDA::CudaScopedTimer>(its, [&]()
        {
            convolveSinglePassSeparateInner(imgSrc,imgDst,d_kernel,KERNEL_RADIUS);
        });
        pth.addMeassurement("convolveSinglePassSeparateInner",st.median);
        checkRes(h_ref,thrust::host_vector<float>(dest));
    }

#endif
    {
        thrust::device_vector<float> d_kernel = h_kernel;
        dest                                  = src;
        auto st                               = Saiga::measureObject<Saiga::CUDA::ScopedTimer>(
            its, [&]() { convolveSinglePassSeparateInner75(imgSrc, imgDst, d_kernel, KERNEL_RADIUS); });
        pth.addMeassurement("convolveSinglePassSeparateInner75", st.median);
        // checkRes(h_ref,thrust::host_vector<float>(dest));
    }

    CUDA_SYNC_CHECK_ERROR();

    {
        thrust::device_vector<float> d_kernel = h_kernel;
        //        dest = src;
        thrust::fill(dest.begin(), dest.end(), 0.0f);
        auto st = Saiga::measureObject<Saiga::CUDA::ScopedTimer>(
            its, [&]() { convolveSinglePassSeparateInnerShuffle(imgSrc, imgDst, d_kernel, KERNEL_RADIUS); });
        pth.addMeassurement("convolveSinglePassSeparateInnerShuffle", st.median);

        thrust::host_vector<char> d(dest);
        ImageView<float> i(h, w, pitch, thrust::raw_pointer_cast(d.data()));
        checkRes(h_imgDst, i);
        //        checkRes2(h_ref,thrust::host_vector<float>(dest));
    }

    CUDA_SYNC_CHECK_ERROR();


#if 1
    {
        dest                                  = src;
        tmp                                   = src;
        thrust::device_vector<float> d_kernel = h_kernel;

        auto st1 = Saiga::measureObject<Saiga::CUDA::ScopedTimer>(
            its, [&]() { convolveRow(imgSrc, imgTmp, d_kernel, KERNEL_RADIUS); });
        pth.addMeassurement("GPU Convolve Separate Row", st1.median);

        auto st2 = Saiga::measureObject<Saiga::CUDA::ScopedTimer>(
            its, [&]() { convolveCol(imgTmp, imgDst, d_kernel, KERNEL_RADIUS); });
        pth.addMeassurement("GPU Convolve Separate Col", st2.median);
        pth.addMeassurement("GPU Convolve Separate Total", st1.median + st2.median);

        //        checkRes(h_ref,thrust::host_vector<float>(dest));
    }
#endif

    {
        auto st = Saiga::measureObject<Saiga::CUDA::ScopedTimer>(its, [&]() {
            cudaMemcpy(thrust::raw_pointer_cast(dest.data()), thrust::raw_pointer_cast(src.data()), N * sizeof(int),
                       cudaMemcpyDeviceToDevice);
        });
        pth.addMeassurement("cudaMemcpy", st.median);
    }
    CUDA_SYNC_CHECK_ERROR();
}

void convolutionTest()
{
    //    convolutionTest2<3>(17,53);
    int w = 2048;
    int h = 1024;
    //    int w = 512;
    //    int h = 256;


    //    convolutionTest2<1>(w,h);
    //    convolutionTest2<2>(w,h);
    //    convolutionTest2<3>(w,h);
    convolutionTest2<4>(w, h);
    //    convolutionTest2<5>(w,h);
    //    convolutionTest2<6>(w,h);
    //    convolutionTest2<7>(w,h);
    //    convolutionTest2<8>(w,h);

    //    convolutionTest2<9>(w,h);
    //    convolutionTest2<10>(w,h);
    //    convolutionTest2<11>(w,h);
    //    convolutionTest2<12>(w,h);
    //    convolutionTest2<13>(w,h);
    //    convolutionTest2<14>(w,h);
    //    convolutionTest2<15>(w,h);
    //    convolutionTest2<16>(w,h);
}

}  // namespace CUDA
}  // namespace Saiga
