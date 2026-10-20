/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/image/all.h"
#include "saiga/core/image/imageGenerator.h"

#include "gtest/gtest.h"

using namespace Saiga;

template <typename T>
static void FillSequence(ImageView<T> img, int offset = 0)
{
    for (int y = 0; y < img.height; ++y)
        for (int x = 0; x < img.width; ++x) img(y, x) = T(offset + y * img.width + x);
}

TEST(ImageView, CopySubViewKeepsSurroundingPixels)
{
    // copyTo used a single memcpy of height * pitch bytes if both views had the same pitch. For sub-image views this
    // also copied the pixels right of the view (and read past the end of the source image).
    TemplatedImage<int> src(4, 8);
    TemplatedImage<int> dst(4, 8);
    FillSequence(src.getImageView());
    dst.getImageView().set(-1);

    auto src_sub = src.getImageView().subImageView(1, 2, 2, 3);
    auto dst_sub = dst.getImageView().subImageView(1, 2, 2, 3);
    ASSERT_EQ(src_sub.pitchBytes, dst_sub.pitchBytes);
    src_sub.copyTo(dst_sub);

    for (int y = 0; y < 4; ++y)
    {
        for (int x = 0; x < 8; ++x)
        {
            bool inside = y >= 1 && y < 3 && x >= 2 && x < 5;
            EXPECT_EQ(dst(y, x), inside ? src(y, x) : -1) << y << " " << x;
        }
    }
}

TEST(ImageView, ScaleDownPow2UnsignedShort)
{
    // The sum was accumulated in unsigned short, which overflowed for bright images.
    TemplatedImage<unsigned short> img(4, 4);
    img.getImageView().set(60000);
    img(0, 0) = 60001;
    img(0, 1) = 60002;
    TemplatedImage<unsigned short> small(2, 2);
    img.getImageView().copyScaleDownPow2(small.getImageView(), 2);
    EXPECT_EQ(small(0, 0), 60001);  // 60000.75 rounded
    EXPECT_EQ(small(1, 1), 60000);
}

TEST(ImageView, ScaleDownPow2UnsignedChar)
{
    // Did not compile, because the unsigned char texel converter had no ZeroFloat().
    TemplatedImage<unsigned char> img(2, 4);
    FillSequence(img.getImageView());
    TemplatedImage<unsigned char> small(1, 2);
    img.getImageView().copyScaleDownPow2(small.getImageView(), 2);
    EXPECT_EQ(small(0, 0), 3);  // (0 + 1 + 4 + 5) / 4 = 2.5 rounded
    EXPECT_EQ(small(0, 1), 5);  // (2 + 3 + 6 + 7) / 4 = 4.5 rounded
}

TEST(ImageView, InterpolateUnsignedShortRounds)
{
    // Interpolated integer values were truncated.
    TemplatedImage<unsigned short> img(1, 2);
    img(0, 0) = 10;
    img(0, 1) = 11;
    auto v    = img.getImageView();
    EXPECT_EQ(v.inter(0, 0.75f), 11);
    EXPECT_EQ(v.inter(0, 0.25f), 10);
}

TEST(ImageView, ArrayImageAndImageArrayView)
{
    // getImageView() and getConstImageView() called themselves through the conversion operators (stack overflow).
    ArrayImage<int> a(3, 2);
    for (int i = 0; i < 6; ++i) a._data[i] = i;
    EXPECT_EQ(a(2, 1), 5);
    EXPECT_EQ(a.getImageView()(1, 0), 2);
    const ArrayImage<int>& ca = a;
    EXPECT_EQ(ca.getConstImageView()(2, 1), 5);
    ImageView<int> converted = a;
    EXPECT_EQ(converted(2, 0), 4);

    std::vector<int> stack(2 * 3 * 2);
    for (size_t i = 0; i < stack.size(); ++i) stack[i] = i;
    ImageArrayView<int> arr(ImageView<int>(3, 2, stack.data()), 2);
    EXPECT_EQ(arr[1](0, 0), 6);
    EXPECT_EQ(arr.atIARVxxx(1, 2, 1), 11);
}

TEST(ImageGenerator, CheckerBoardNonSquare)
{
    // x and y were swapped, which wrote outside of the image for numQuadsX > numQuadsY and left parts of the image
    // unset. With an odd numQuadsY, quads in the same row also had the same color.
    vec3 c1(1, 1, 1), c2(0, 0, 0);
    for (auto [nx, ny] : {std::pair{4, 1}, std::pair{1, 4}, std::pair{3, 3}, std::pair{2, 3}})
    {
        auto img = ImageGenerator::checkerBoard(c1, c2, 2, nx, ny);
        ASSERT_EQ(img->width, 2 * nx);
        ASSERT_EQ(img->height, 2 * ny);
        for (int y = 0; y < img->height; ++y)
        {
            for (int x = 0; x < img->width; ++x)
            {
                bool first          = ((x / 2) + (y / 2)) % 2 == 0;
                unsigned char value = first ? 255 : 0;
                EXPECT_EQ(img->at<ucvec3>(y, x), ucvec3(value, value, value))
                    << nx << "x" << ny << " pixel " << y << " " << x;
            }
        }
    }
}
