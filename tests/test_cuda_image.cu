/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "saiga/core/framework/framework.h"
#include "saiga/core/image/all.h"
#include "saiga/cuda/imageProcessing/NppiHelper.h"
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

        context = NPPI::CreateStreamContextWithStream(0);
    }


    TemplatedImage<unsigned char> image_1;
    TemplatedImage<ucvec4> image_4;

    CUDA::CudaImage<unsigned char> d_image_1;
    CUDA::CudaImage<ucvec4> d_image_4;

    SaigaNppStreamContext context;
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
}

TEST(CudaImage, GaussFilter)
{
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
}

}  // namespace Saiga

int main()
{
    Saiga::initSaigaSampleNoWindow();
    testing::InitGoogleTest();

    return RUN_ALL_TESTS();
}
