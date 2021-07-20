/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/cuda/imageProcessing/NppiHelper.h"
//
#include "saiga/core/framework/framework.h"
#include "saiga/core/image/all.h"
#include "saiga/cuda/CudaInfo.h"
#include "saiga/cuda/imageProcessing/image.h"

#include "gtest/gtest.h"

#include "compare_numbers.h"

namespace Saiga
{ 
class CudaImageTest
{
   public:
    CudaImageTest()
    {
        TemplatedImage<ucvec3> image_3;
        image_3.load("textures/redie.png");
        image_1.create(image_3.dimensions());
        image_4.create(image_3.dimensions());

        ImageTransformation::addAlphaChannel(image_3.getImageView(), image_4.getImageView());
        ImageTransformation::RGBAToGray8(image_4.getImageView(), image_1.getImageView());

        image_1.save("gray.png");
        image_4.save("rgba.png");

#ifdef SAIGA_USE_CUDA_TOOLKIT
        context = NPPI::CreateStreamContextWithStream(0);
#endif
    }

    TemplatedImage<unsigned char> image_1;
    TemplatedImage<ucvec4> image_4;

    CUDA::CudaImage<unsigned char> d_image_1;
    CUDA::CudaImage<ucvec4> d_image_4;
#ifdef SAIGA_USE_CUDA_TOOLKIT
    SaigaNppStreamContext context;
#endif
};

std::unique_ptr<CudaImageTest> test;

TEST(CudaImage, UploadDownload)
{
    test = std::make_unique<CudaImageTest>();

    // with explicit upload download method
    test->d_image_4.upload(test->image_4.getConstImageView());
    TemplatedImage<ucvec4> result(test->image_4.dimensions());
    result.makeZero();
    test->d_image_4.download(result.getImageView());
    EXPECT_EQ(result.getImageView(), test->image_4.getImageView());

    // with constructors
    result.makeZero();
    test->d_image_4.clear();
    test->d_image_4 = CUDA::CudaImage<ucvec4>(test->image_4.getImageView());
    test->d_image_4.download(result.getImageView());
    EXPECT_EQ(result.getImageView(), test->image_4.getImageView());

    test->d_image_1.upload(test->image_1.getConstImageView());
    test->d_image_4.upload(test->image_4.getConstImageView());
}

TEST(CudaImage, Resize)
{
#ifdef SAIGA_USE_CUDA_TOOLKIT
    CUDA::CudaImage<unsigned char> img_small;
    TemplatedImage<unsigned char> result;

    img_small.create(test->d_image_1.h * 0.5, test->d_image_1.w * 0.5);
    NPPI::ResizeLinear(test->d_image_1.getConstImageView(), img_small.getImageView(), test->context);
    img_small.download(result);
    result.save("gray_small.png");

    img_small.create(test->d_image_1.h * 2, test->d_image_1.w * 2);
    NPPI::ResizeLinear(test->d_image_1.getConstImageView(), img_small.getImageView(), test->context);
    img_small.download(result);
    result.save("gray_large.png");

    img_small.create(test->d_image_1.h * 0.133, test->d_image_1.w * 0.133);
    NPPI::ResizeLinear(test->d_image_1.getConstImageView(), img_small.getImageView(), test->context);
    img_small.download(result);
    result.save("gray_very_small.png");
#endif
}

TEST(CudaImage, GaussFilter)
{
#ifdef SAIGA_USE_CUDA_TOOLKIT
    CUDA::CudaImage<unsigned char> filtered_image;
    CUDA::CudaImage<unsigned char> cpy_filtered_image;
    TemplatedImage<unsigned char> result;

    filtered_image.create(test->d_image_1.h, test->d_image_1.w);
    NPPI::GaussFilter(test->d_image_1.getConstImageView(), filtered_image.getImageView(), test->context);
    filtered_image.download(result);
    result.save("gray_gauss.png");

    cpy_filtered_image = filtered_image;
    NPPI::GaussFilter(filtered_image.getConstImageView(), cpy_filtered_image.getImageView(), test->context);
    NPPI::GaussFilter(cpy_filtered_image.getConstImageView(), filtered_image.getImageView(), test->context);
    filtered_image.download(result);
    result.save("gray_very_gauss.png");
#endif
}

__global__ static void ReadValue(ImageView<unsigned char> image, int row, int col, ArrayView<int> target)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= 1) return;

    int v     = image(row, col);
    target[0] = v;
}

texture<unsigned char, cudaTextureType2D, cudaReadModeElementType> texture_reference;

__global__ static void ReadValueTexRef(int w, int h, int row, int col, ArrayView<int> target)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= 1) return;

    float u = float(col) / w;
    float v = float(row) / h;
    float f = tex2D(texture_reference, u, v);

    printf("%f,%f,%f\n", u, v, f);

    //    int val   = tex2D(texture_reference, float(col) / w, float(row) / h);
    int val   = tex2D(texture_reference, col, row);
    target[0] = val;
}

__global__ static void ReadValueTexObj(cudaTextureObject_t texObj, int w, int h, int row, int col,
                                       ArrayView<int> target)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= 1) return;

    float u = float(col) / w;
    float v = float(row) / h;
    float f = tex2D<unsigned char>(texObj, u, v);

    printf("%f,%f,%f\n", u, v, f);

    //    int val   = tex2D(texture_reference, float(col) / w, float(row) / h);
    int val   = tex2D<unsigned char>(texObj, col + 0.9, row);
    target[0] = val;
}

TEST(CudaImage, ImageAccess)
{
    int r     = 67;
    int c     = 180;
    int value = test->image_1(r, c);

    {
        // Test if reading from a simple cuda kernel works.
        thrust::device_vector<int> target(1);
        ReadValue<<<1, 1>>>(test->d_image_1.getImageView(), r, c, target);
        int value2 = target[0];
        EXPECT_EQ(value, value2);
    }

    {
        // Texture reference

        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();
        size_t offset;
        CHECK_CUDA_ERROR(cudaBindTexture2D(&offset, texture_reference, test->d_image_1.data(), channelDesc,
                                           test->d_image_1.cols, test->d_image_1.rows, test->d_image_1.pitchBytes));

        thrust::device_vector<int> target(1);
        ReadValueTexRef<<<1, 1>>>(test->d_image_1.cols, test->d_image_1.rows, r, c, target);
        int value2 = target[0];
        EXPECT_EQ(value, value2);
    }

    {
        // Texture object

        auto img = test->d_image_1.getImageView();

        auto texObj = test->d_image_1.GetTextureObject();

        thrust::device_vector<int> target(1);
        ReadValueTexObj<<<1, 1>>>(texObj, test->d_image_1.cols, test->d_image_1.rows, r, c, target);
        int value2 = target[0];
        EXPECT_EQ(value, value2);
    }
}

}  // namespace Saiga

int main()
{
#ifdef SAIGA_NPPI_HAS_STREAM_CONTEXT
    std::cout << "NPPI stream context found!" << std::endl;
#else
    std::cout << "NPPI stream context not found!" << std::endl;
#endif
    Saiga::CUDA::initCUDA();
    Saiga::CUDA::printCUDAInfo();

    Saiga::initSaigaSampleNoWindow();
    testing::InitGoogleTest();

    return RUN_ALL_TESTS();
}
