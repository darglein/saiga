/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/image/all.h"
#include "saiga/core/math/random.h"
#include "saiga/cuda/CudaInfo.h"
#include "saiga/cuda/imageProcessing/image.h"
#include "saiga/cuda/imageProcessing/imageProcessing.h"

#include "gtest/gtest.h"

#include <functional>

namespace Saiga
{
static TemplatedImage<float> RandomImage(int h, int w, float min_value = -1, float max_value = 1)
{
    TemplatedImage<float> img(h, w);
    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            img(y, x) = Random::sampleDouble(min_value, max_value);
        }
    }
    return img;
}

static TemplatedImage<float> Download(CUDA::CudaImage<float>& d_img)
{
    TemplatedImage<float> img(d_img.h, d_img.w);
    d_img.download(img.getImageView());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    return img;
}

static void ExpectImageNear(ImageView<const float> expected, ImageView<const float> actual, float eps,
                            const std::string& name)
{
    ASSERT_EQ(expected.dimensions(), actual.dimensions()) << name;
    int errors = 0;
    for (int y = 0; y < expected.h; ++y)
    {
        for (int x = 0; x < expected.w; ++x)
        {
            float e = expected(y, x);
            float a = actual(y, x);
            if (!(std::abs(e - a) <= eps * std::max(1.f, std::abs(e))))
            {
                if (errors < 5)
                {
                    ADD_FAILURE() << name << ": pixel (" << y << "," << x << ") expected " << e << " got " << a;
                }
                errors++;
            }
        }
    }
    EXPECT_EQ(errors, 0) << name << " (" << expected.h << "x" << expected.w << ")";
}

// Correlation along the rows with clamp-to-edge border handling.
static TemplatedImage<float> CpuConvolveRow(ImageView<const float> src, const std::vector<float>& kernel)
{
    int radius = kernel.size() / 2;
    TemplatedImage<float> dst(src.h, src.w);
    for (int y = 0; y < src.h; ++y)
    {
        for (int x = 0; x < src.w; ++x)
        {
            double sum = 0;
            for (int k = -radius; k <= radius; ++k)
            {
                int sx = std::min(std::max(x + k, 0), src.w - 1);
                sum += kernel[k + radius] * src(y, sx);
            }
            dst(y, x) = sum;
        }
    }
    return dst;
}

static TemplatedImage<float> CpuConvolveCol(ImageView<const float> src, const std::vector<float>& kernel)
{
    int radius = kernel.size() / 2;
    TemplatedImage<float> dst(src.h, src.w);
    for (int y = 0; y < src.h; ++y)
    {
        for (int x = 0; x < src.w; ++x)
        {
            double sum = 0;
            for (int k = -radius; k <= radius; ++k)
            {
                int sy = std::min(std::max(y + k, 0), src.h - 1);
                sum += kernel[k + radius] * src(sy, x);
            }
            dst(y, x) = sum;
        }
    }
    return dst;
}

// An asymmetric kernel, so that a flipped kernel or swapped axes are detected.
static std::vector<float> RandomKernel(int radius)
{
    std::vector<float> kernel(radius * 2 + 1);
    float sum = 0;
    for (auto& k : kernel)
    {
        k = Random::sampleDouble(0.1, 1);
        sum += k;
    }
    for (auto& k : kernel) k /= sum;
    return kernel;
}

static const std::vector<std::pair<int, int>> test_sizes = {{48, 64}, {53, 97}, {20, 30}, {1, 37}, {131, 1}, {67, 258}};

using SinglePassFunction = std::function<void(ImageView<float>, ImageView<float>, Saiga::ArrayView<float>, int)>;

static void TestSeparable(const std::string& name, SinglePassFunction f, std::vector<int> radii)
{
    for (auto size : test_sizes)
    {
        auto h_src = RandomImage(size.first, size.second);
        CUDA::CudaImage<float> d_src(h_src.getConstImageView());
        CUDA::CudaImage<float> d_dst(h_src.h, h_src.w);

        for (int radius : radii)
        {
            auto kernel = RandomKernel(radius);
            thrust::device_vector<float> d_kernel(kernel);

            auto expected =
                CpuConvolveCol(CpuConvolveRow(h_src.getConstImageView(), kernel).getConstImageView(), kernel);

            CUDA::fill(d_dst.getImageView(), -12345);
            f(d_src.getImageView(), d_dst.getImageView(), d_kernel, radius);
            CHECK_CUDA_ERROR(cudaDeviceSynchronize());
            auto result = Download(d_dst);
            ExpectImageNear(expected.getConstImageView(), result.getConstImageView(), 1e-5,
                            name + " radius " + std::to_string(radius));
        }
    }
}

TEST(CudaConvolution, Row)
{
    std::vector<int> radii;
    for (int r = 1; r <= SAIGA_MAX_CONVOLUTION_RADIUS; ++r) radii.push_back(r);

    for (auto size : test_sizes)
    {
        auto h_src = RandomImage(size.first, size.second);
        CUDA::CudaImage<float> d_src(h_src.getConstImageView());
        CUDA::CudaImage<float> d_dst(h_src.h, h_src.w);
        for (int radius : radii)
        {
            auto kernel = RandomKernel(radius);
            thrust::device_vector<float> d_kernel(kernel);
            CUDA::convolveRow(d_src.getImageView(), d_dst.getImageView(), d_kernel, radius);
            auto expected = CpuConvolveRow(h_src.getConstImageView(), kernel);
            auto result   = Download(d_dst);
            ExpectImageNear(expected.getConstImageView(), result.getConstImageView(), 1e-5,
                            "row radius " + std::to_string(radius));
        }
    }
}

TEST(CudaConvolution, Col)
{
    std::vector<int> radii;
    for (int r = 1; r <= SAIGA_MAX_CONVOLUTION_RADIUS; ++r) radii.push_back(r);

    for (auto size : test_sizes)
    {
        auto h_src = RandomImage(size.first, size.second);
        CUDA::CudaImage<float> d_src(h_src.getConstImageView());
        CUDA::CudaImage<float> d_dst(h_src.h, h_src.w);
        for (int radius : radii)
        {
            auto kernel = RandomKernel(radius);
            thrust::device_vector<float> d_kernel(kernel);
            CUDA::convolveCol(d_src.getImageView(), d_dst.getImageView(), d_kernel, radius);
            auto expected = CpuConvolveCol(h_src.getConstImageView(), kernel);
            auto result   = Download(d_dst);
            ExpectImageNear(expected.getConstImageView(), result.getConstImageView(), 1e-5,
                            "col radius " + std::to_string(radius));
        }
    }
}

TEST(CudaConvolution, SinglePassOuterLinear)
{
    TestSeparable("OuterLinear", CUDA::convolveSinglePassSeparateOuterLinear, {1, 2, 3, 4, 5, 6, 7, 8, 9});
}

TEST(CudaConvolution, SinglePassOuterHalo)
{
    TestSeparable("OuterHalo", CUDA::convolveSinglePassSeparateOuterHalo, {1, 2, 3, 4, 5, 6, 7, 8});
}

TEST(CudaConvolution, SinglePassInner)
{
    TestSeparable("Inner", CUDA::convolveSinglePassSeparateInner, {1, 2, 3, 4, 5, 6, 7, 8});
}

TEST(CudaConvolution, SinglePassInner75)
{
    TestSeparable("Inner75", CUDA::convolveSinglePassSeparateInner75, {1, 2, 3, 4, 5, 6, 7, 8});
}

TEST(CudaImageProcessing, PointOperations)
{
    auto h_src = RandomImage(37, 133);
    CUDA::CudaImage<float> d_img(h_src.getConstImageView());

    TemplatedImage<float> expected(h_src.h, h_src.w);

    CUDA::add(d_img.getImageView(), 2.5f);
    for (int y = 0; y < h_src.h; ++y)
        for (int x = 0; x < h_src.w; ++x) expected(y, x) = h_src(y, x) + 2.5f;
    ExpectImageNear(expected.getConstImageView(), Download(d_img).getConstImageView(), 1e-6, "add");

    CUDA::mult(d_img.getImageView(), -3.f);
    for (int y = 0; y < h_src.h; ++y)
        for (int x = 0; x < h_src.w; ++x) expected(y, x) *= -3.f;
    ExpectImageNear(expected.getConstImageView(), Download(d_img).getConstImageView(), 1e-6, "mult");

    CUDA::abs(d_img.getImageView());
    for (int y = 0; y < h_src.h; ++y)
        for (int x = 0; x < h_src.w; ++x) expected(y, x) = std::abs(expected(y, x));
    ExpectImageNear(expected.getConstImageView(), Download(d_img).getConstImageView(), 1e-6, "abs");

    CUDA::fill(d_img.getImageView(), 7.f);
    for (int y = 0; y < h_src.h; ++y)
        for (int x = 0; x < h_src.w; ++x) expected(y, x) = 7.f;
    ExpectImageNear(expected.getConstImageView(), Download(d_img).getConstImageView(), 0, "fill");
}

TEST(CudaImageProcessing, Subtract)
{
    auto h_a = RandomImage(29, 141);
    auto h_b = RandomImage(29, 141);
    CUDA::CudaImage<float> d_a(h_a.getConstImageView());
    CUDA::CudaImage<float> d_b(h_b.getConstImageView());
    CUDA::CudaImage<float> d_dst(h_a.h, h_a.w);

    CUDA::subtract(d_a.getImageView(), d_b.getImageView(), d_dst.getImageView());

    TemplatedImage<float> expected(h_a.h, h_a.w);
    for (int y = 0; y < h_a.h; ++y)
        for (int x = 0; x < h_a.w; ++x) expected(y, x) = h_a(y, x) - h_b(y, x);
    ExpectImageNear(expected.getConstImageView(), Download(d_dst).getConstImageView(), 0, "subtract");
}

TEST(CudaImageProcessing, SubtractMulti)
{
    const int n = 4, h = 31, w = 139;
    std::vector<TemplatedImage<float>> h_src;
    for (int i = 0; i < n; ++i) h_src.push_back(RandomImage(h, w));

    // n images stored consecutively with the same pitch
    CUDA::CudaImage<float> d_src(h * n, w);
    CUDA::CudaImage<float> d_dst(h * (n - 1), w);
    ImageView<float> src_view = d_src.getImageView();
    src_view.h                = h;
    ImageView<float> dst_view = d_dst.getImageView();
    dst_view.h                = h;
    ImageArrayView<float> src(src_view, n);
    ImageArrayView<float> dst(dst_view, n - 1);
    for (int i = 0; i < n; ++i)
    {
        CUDA::CopyImage(h_src[i].getConstImageView(), src[i], cudaMemcpyHostToDevice);
    }

    CUDA::subtractMulti(src, dst);

    for (int i = 0; i < n - 1; ++i)
    {
        TemplatedImage<float> result(h, w);
        CUDA::CopyImage(ImageView<const float>(dst[i]), result.getImageView(), cudaMemcpyDeviceToHost);
        TemplatedImage<float> expected(h, w);
        for (int y = 0; y < h; ++y)
            for (int x = 0; x < w; ++x) expected(y, x) = h_src[i + 1](y, x) - h_src[i](y, x);
        ExpectImageNear(expected.getConstImageView(), result.getConstImageView(), 0,
                        "subtractMulti " + std::to_string(i));
    }
}

TEST(CudaImageProcessing, Conversions)
{
    const int h = 23, w = 157;
    std::vector<uchar3> rgb_data(h * w), rgb2_data(h * w);
    std::vector<uchar4> rgba_data(h * w);
    for (auto& c : rgb_data)
        c = make_uchar3(Random::uniformInt(0, 255), Random::uniformInt(0, 255), Random::uniformInt(0, 255));
    ImageView<uchar3> rgb(h, w, rgb_data.data());
    ImageView<uchar3> rgb2(h, w, rgb2_data.data());
    ImageView<uchar4> rgba(h, w, rgba_data.data());

    CUDA::CudaImage<uchar3> d_rgb{ImageView<const uchar3>(rgb)};
    CUDA::CudaImage<uchar4> d_rgba(h, w);
    CUDA::CudaImage<uchar3> d_rgb2(h, w);
    CUDA::CudaImage<float> d_gray(h, w);

    CUDA::convertRGBtoRGBA(d_rgb.getImageView(), d_rgba.getImageView(), 17);
    d_rgba.download(rgba);
    CUDA::convertRGBAtoRGB(d_rgba.getImageView(), d_rgb2.getImageView());
    d_rgb2.download(rgb2);
    CUDA::convertRGBAtoGrayscale(d_rgba.getImageView(), d_gray.getImageView());
    auto gray = Download(d_gray);

    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            auto c = rgb(y, x);
            auto r = rgba(y, x);
            EXPECT_TRUE(r.x == c.x && r.y == c.y && r.z == c.z && r.w == 17);
            auto r2 = rgb2(y, x);
            EXPECT_TRUE(r2.x == c.x && r2.y == c.y && r2.z == c.z);
            EXPECT_NEAR(gray(y, x), 0.2126f * c.x + 0.7152f * c.y + 0.0722f * c.z, 1e-3);
        }
    }

    CUDA::convertBGRtoRGBA(d_rgb.getImageView(), d_rgba.getImageView(), 3);
    d_rgba.download(rgba);
    CUDA::convertRGBAtoBGR(d_rgba.getImageView(), d_rgb2.getImageView());
    d_rgb2.download(rgb2);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    for (int y = 0; y < h; ++y)
    {
        for (int x = 0; x < w; ++x)
        {
            auto c = rgb(y, x);
            auto r = rgba(y, x);
            EXPECT_TRUE(r.x == c.z && r.y == c.y && r.z == c.x && r.w == 3);
            auto r2 = rgb2(y, x);
            EXPECT_TRUE(r2.x == c.x && r2.y == c.y && r2.z == c.z);
        }
    }
}

}  // namespace Saiga

int main(int argc, char** argv)
{
    Saiga::CUDA::initCUDA();
    Saiga::Random::setSeed(3487);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
