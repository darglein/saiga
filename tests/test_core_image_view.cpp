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

TEST(ImageView, CopyContiguousAndDifferentPitch)
{
    TemplatedImage<unsigned char> src(3, 5);
    FillSequence(src.getImageView());

    // Same (padded) pitch
    TemplatedImage<unsigned char> dst1(3, 5);
    src.getImageView().copyTo(dst1.getImageView());
    EXPECT_EQ(src.getConstImageView(), dst1.getConstImageView());

    // Tightly packed destination with a different pitch
    std::vector<unsigned char> packed(15, 0);
    ImageView<unsigned char> dst2(3, 5, packed.data());
    src.getImageView().copyTo(dst2);
    for (int i = 0; i < 15; ++i) EXPECT_EQ(packed[i], i);

    // Converting copy
    TemplatedImage<float> dst3(3, 5);
    src.getImageView().copyTo(dst3.getImageView());
    EXPECT_EQ(dst3(2, 4), 14.f);

    TemplatedImage<float> dst4(3, 5);
    src.getImageView().copyTo(dst4.getImageView(), 0.5f);
    EXPECT_EQ(dst4(2, 4), 7.f);
}

TEST(ImageView, CopyYFlippedView)
{
    TemplatedImage<int> src(3, 4);
    FillSequence(src.getImageView());
    TemplatedImage<int> dst(3, 4);
    src.getImageView().yFlippedImageView().copyTo(dst.getImageView());
    for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 4; ++x) EXPECT_EQ(dst(y, x), src(2 - y, x));
}

TEST(ImageView, SubImageViewAndSetSubImage)
{
    TemplatedImage<int> img(5, 6);
    FillSequence(img.getImageView());
    auto sub = img.getImageView().subImageView(2, 3, 2, 2);
    EXPECT_EQ(sub.h, 2);
    EXPECT_EQ(sub.w, 2);
    EXPECT_EQ(sub(0, 0), img(2, 3));
    EXPECT_EQ(sub(1, 1), img(3, 4));

    auto crop = img.getImageView().centerCrop(3, 2);
    EXPECT_EQ(crop(0, 0), img(1, 2));
    auto crop2 = img.getImageView().centerCrop2(1, 2);
    EXPECT_EQ(crop2.h, 3);
    EXPECT_EQ(crop2.w, 2);
    EXPECT_EQ(crop2(0, 0), img(1, 2));

    TemplatedImage<int> patch(2, 3);
    patch.getImageView().set(-7);
    img.getImageView().setSubImage(3, 3, patch.getImageView());
    EXPECT_EQ(img(3, 3), -7);
    EXPECT_EQ(img(4, 5), -7);
    EXPECT_EQ(img(2, 3), 2 * 6 + 3);
    EXPECT_EQ(img(3, 2), 3 * 6 + 2);
}

TEST(ImageView, FlipAndIterate)
{
    TemplatedImage<int> img(3, 3);
    FillSequence(img.getImageView());
    img.getImageView().flipX();
    EXPECT_EQ(img(0, 0), 2);
    EXPECT_EQ(img(0, 1), 1);
    img.getImageView().flipY();
    EXPECT_EQ(img(0, 0), 8);
    EXPECT_EQ(img(2, 2), 0);

    int count = 0;
    for (auto row : img.getImageView())
    {
        for (auto p : row)
        {
            EXPECT_EQ(p.value(), img(p.y(), p.x()));
            ++count;
        }
    }
    EXPECT_EQ(count, 9);
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

TEST(ImageView, ScaleLinearAndMedian)
{
    TemplatedImage<float> img(2, 2);
    img.getImageView().set(3.f);
    TemplatedImage<float> big(4, 4);
    img.getImageView().copyScaleLinear(big.getImageView());
    for (int y = 0; y < 4; ++y)
        for (int x = 0; x < 4; ++x) EXPECT_FLOAT_EQ(big(y, x), 3.f);

    TemplatedImage<int> m(2, 4);
    m(0, 0) = 1;
    m(0, 1) = 9;
    m(1, 0) = 5;
    m(1, 1) = 3;
    m(0, 2) = 4;
    m(0, 3) = 4;
    m(1, 2) = 8;
    m(1, 3) = 0;
    TemplatedImage<int> med(1, 2);
    m.getImageView().copyToScaleDownMedian(med.getImageView());
    EXPECT_EQ(med(0, 0), 3);
    EXPECT_EQ(med(0, 1), 4);
    m.getImageView().copyToScaleDownMedian<int, false>(med.getImageView());
    EXPECT_EQ(med(0, 0), 5);

    m.getImageView().copyToScaleDownEverySecond(med.getImageView());
    EXPECT_EQ(med(0, 0), 1);
    EXPECT_EQ(med(0, 1), 4);
}

TEST(ImageView, Gradients)
{
    TemplatedImage<float> img(3, 4);
    for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 4; ++x) img(y, x) = x * x + 10 * y;
    TemplatedImage<float> g(3, 4);
    img.getImageView().gx(g.getImageView());
    EXPECT_FLOAT_EQ(g(0, 0), 1);
    EXPECT_FLOAT_EQ(g(1, 1), 2);
    EXPECT_FLOAT_EQ(g(1, 2), 4);
    EXPECT_FLOAT_EQ(g(2, 3), 5);
    img.getImageView().gy(g.getImageView());
    for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 4; ++x) EXPECT_FLOAT_EQ(g(y, x), 10);
}

TEST(ImageView, BorderAccess)
{
    TemplatedImage<int> img(2, 3);
    FillSequence(img.getImageView());
    auto v = img.getImageView();
    EXPECT_EQ(v.clampedRead(-5, 10), img(0, 2));
    EXPECT_EQ(v.borderRead(2, 0, -1), -1);
    EXPECT_EQ(v.borderRead(1, 1, -1), 4);
    v.clampedWrite(-1, 0, 100);
    v.clampedWrite(0, 3, 100);
    for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 3; ++x) EXPECT_EQ(img(y, x), y * 3 + x);
    v.clampedWrite(1, 2, 100);
    EXPECT_EQ(img(1, 2), 100);
    int y = -1, x = 4;
    v.mirrorToEdge(y, x);
    EXPECT_EQ(y, 1);
    EXPECT_EQ(x, 0);
    EXPECT_TRUE(v.inImage(1, 2));
    EXPECT_FALSE(v.inImage(2, 0));
    EXPECT_FALSE(v.inImage2(0, -1));
}

TEST(ImageView, FindMinMaxOutlierAndFinite)
{
    TemplatedImage<float> img(1, 4);
    img(0, 0) = 0;
    img(0, 1) = -2;
    img(0, 2) = 5;
    img(0, 3) = 3;
    float mi, ma;
    img.getImageView().findMinMaxOutlier(mi, ma, 0.f);
    EXPECT_EQ(mi, -2);
    EXPECT_EQ(ma, 5);
    EXPECT_TRUE(img.getImageView().isFinite());
    img(0, 3) = std::numeric_limits<float>::infinity();
    EXPECT_FALSE(img.getImageView().isFinite());
}

TEST(ImageView, ChannelOps)
{
    TemplatedImage<ucvec3> img(2, 2);
    img.getImageView().set(ucvec3(1, 2, 3));
    img.getImageView().swapChannels(0, 2);
    EXPECT_EQ(img(1, 1), ucvec3(3, 2, 1));
    img.getImageView().setChannel(1, 9);
    EXPECT_EQ(img(0, 1), ucvec3(3, 9, 1));
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

TEST(ImageTransformation, DepthToRGBAUnsignedShortClamps)
{
    // Depths outside of [minD,maxD] were not clamped and wrapped around in the 8 bit output.
    TemplatedImage<uint16_t> depth(1, 4);
    depth(0, 0) = 100;
    depth(0, 1) = 150;
    depth(0, 2) = 200;
    depth(0, 3) = 400;
    TemplatedImage<ucvec4> rgba(1, 4);
    ImageTransformation::depthToRGBA(depth.getConstImageView(), rgba.getImageView(), uint16_t(100), uint16_t(200));
    EXPECT_EQ(rgba(0, 0), ucvec4(0, 0, 0, 255));
    EXPECT_EQ(rgba(0, 1)(0), 127);
    EXPECT_EQ(rgba(0, 2), ucvec4(255, 255, 255, 255));
    EXPECT_EQ(rgba(0, 3), ucvec4(255, 255, 255, 255));

    depth(0, 3) = 50;
    ImageTransformation::depthToRGBA(depth.getConstImageView(), rgba.getImageView(), uint16_t(100), uint16_t(200));
    EXPECT_EQ(rgba(0, 3), ucvec4(0, 0, 0, 255));
}

TEST(ImageTransformation, DepthToRGBAFloat)
{
    TemplatedImage<float> depth(1, 3);
    depth(0, 0) = -1;
    depth(0, 1) = 0.5;
    depth(0, 2) = 3;
    TemplatedImage<ucvec4> rgba(1, 3);
    ImageTransformation::depthToRGBA(depth.getConstImageView(), rgba.getImageView(), 0, 1);
    EXPECT_EQ(rgba(0, 0), ucvec4(0, 0, 0, 255));
    EXPECT_EQ(rgba(0, 1)(1), 127);
    EXPECT_EQ(rgba(0, 2), ucvec4(255, 255, 255, 255));

    ImageTransformation::depthToRGBA_HSV(depth.getConstImageView(), rgba.getImageView(), 0, 1);
    EXPECT_EQ(rgba(0, 0)(3), 255);
}

TEST(ImageTransformation, SharpnessIsSymmetric)
{
    // Only rising edges were counted (max of the signed differences), so a mirrored image was "less sharp".
    TemplatedImage<unsigned char> img(5, 6);
    for (int y = 0; y < 5; ++y)
        for (int x = 0; x < 6; ++x) img(y, x) = x < 3 ? 0 : 200;
    float s1 = ImageTransformation::sharpness(img.getConstImageView());
    EXPECT_GT(s1, 0);
    img.getImageView().flipX();
    float s2 = ImageTransformation::sharpness(img.getConstImageView());
    EXPECT_FLOAT_EQ(s1, s2);
}

TEST(ImageTransformation, ChannelConversions)
{
    TemplatedImage<ucvec3> rgb(2, 3);
    rgb.getImageView().set(ucvec3(10, 20, 30));
    TemplatedImage<ucvec4> rgba(2, 3);
    ImageTransformation::addAlphaChannel(rgb.getConstImageView(), rgba.getImageView(), 77);
    EXPECT_EQ(rgba(1, 2), ucvec4(10, 20, 30, 77));

    TemplatedImage<ucvec3> rgb2(2, 3);
    ImageTransformation::RemoveAlphaChannel(rgba.getConstImageView(), rgb2.getImageView());
    EXPECT_EQ(rgb2(1, 2), ucvec3(10, 20, 30));

    TemplatedImage<unsigned char> gray(2, 3);
    ImageTransformation::RGBAToGray8(rgba.getConstImageView(), gray.getImageView());
    // 0.299 * 10 + 0.587 * 20 + 0.114 * 30 = 18.15
    EXPECT_EQ(gray(0, 0), 18);

    TemplatedImage<float> grayf(2, 3);
    ImageTransformation::RGBAToGrayF(rgba.getConstImageView(), grayf.getImageView(), 1.f / 255);
    EXPECT_NEAR(grayf(0, 0), 18.15f / 255, 1e-5);

    ImageTransformation::Gray8ToRGBA(gray.getImageView(), rgba.getImageView(), 5);
    EXPECT_EQ(rgba(0, 0), ucvec4(18, 18, 18, 5));
    ImageTransformation::Gray8ToRGB(gray.getImageView(), rgb.getImageView());
    EXPECT_EQ(rgb(0, 0), ucvec3(18, 18, 18));
}

TEST(ImageTransformation, GrayOfWhiteIsWhite)
{
    TemplatedImage<ucvec4> rgba(1, 1);
    rgba(0, 0) = ucvec4(255, 255, 255, 255);
    TemplatedImage<unsigned char> gray(1, 1);
    ImageTransformation::RGBAToGray8(rgba.getConstImageView(), gray.getImageView());
    EXPECT_EQ(gray(0, 0), 255);
}

TEST(ImageTransformation, FloatConversions)
{
    TemplatedImage<float> img(1, 3);
    img(0, 0) = -1;
    img(0, 1) = 0.5f;
    img(0, 2) = 2;
    auto c    = ImageTransformation::FloatToChar(img.getImageView());
    EXPECT_EQ(c(0, 0), 0);
    EXPECT_EQ(c(0, 1), 128);
    EXPECT_EQ(c(0, 2), 255);
    auto s = ImageTransformation::FloatToShort(img.getImageView());
    EXPECT_EQ(s(0, 0), 0);
    EXPECT_EQ(s(0, 1), 32768);
    EXPECT_EQ(s(0, 2), 65535);
    auto turbo = ImageTransformation::ColorizeTurbo(img.getImageView());
    EXPECT_EQ(turbo.dimensions(), img.dimensions());
}

TEST(ImageTransformation, ScaleDownAndDifferences)
{
    TemplatedImage<ucvec4> src(2, 4);
    for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 4; ++x) src(y, x) = ucvec4(x * 10, y * 10, 255, 0);
    TemplatedImage<ucvec4> dst(1, 2);
    ImageTransformation::ScaleDown2(src.getConstImageView(), dst.getImageView());
    EXPECT_EQ(dst(0, 0), ucvec4(5, 5, 255, 0));
    EXPECT_EQ(dst(0, 1), ucvec4(25, 5, 255, 0));

    TemplatedImage<ucvec3> a(1, 2), b(1, 2);
    a(0, 0)  = ucvec3(10, 20, 30);
    a(0, 1)  = ucvec3(0, 0, 0);
    b(0, 0)  = ucvec3(15, 10, 30);
    b(0, 1)  = ucvec3(0, 0, 0);
    auto err = ImageTransformation::AbsolutePixelError(a.getConstImageView(), b.getConstImageView());
    EXPECT_EQ(err(0, 0), 10);
    EXPECT_EQ(err(0, 1), 0);
    EXPECT_EQ(ImageTransformation::L1Difference(a.getConstImageView(), b.getConstImageView()), 15);
    auto error_img = ImageTransformation::ErrorImage(a.getImageView(), b.getImageView());
    EXPECT_EQ(error_img.dimensions(), a.dimensions());

    TemplatedImage<unsigned char> g1(1, 2), g2(1, 2);
    g1(0, 0)  = 3;
    g1(0, 1)  = 200;
    g2(0, 0)  = 10;
    g2(0, 1)  = 100;
    auto gerr = ImageTransformation::AbsolutePixelError(g1.getConstImageView(), g2.getConstImageView());
    EXPECT_EQ(gerr(0, 0), 7);
    EXPECT_EQ(gerr(0, 1), 100);
    EXPECT_EQ(ImageTransformation::L1Difference(g1.getConstImageView(), g2.getConstImageView()), 107);
}

TEST(ImageDraw, Shapes)
{
    TemplatedImage<unsigned char> img(8, 10);
    img.makeZero();
    // Partially outside of the image: must not write out of bounds
    ImageDraw::FillRectangle(img.getImageView(), ivec2(-2, -2), ivec2(2, 3), (unsigned char)1);
    EXPECT_EQ(img(0, 0), 1);
    EXPECT_EQ(img(2, 1), 1);
    EXPECT_EQ(img(3, 1), 0);
    EXPECT_EQ(img(0, 2), 0);

    img.makeZero();
    ImageDraw::DrawRectangle(img.getImageView(), ivec2(2, 1), ivec2(6, 5), (unsigned char)2, 1);
    EXPECT_EQ(img(1, 2), 2);
    EXPECT_EQ(img(1, 5), 2);
    EXPECT_EQ(img(4, 2), 2);
    EXPECT_EQ(img(4, 5), 2);
    EXPECT_EQ(img(2, 3), 0);
    EXPECT_EQ(img(1, 6), 0);

    img.makeZero();
    ImageDraw::drawLineBresenham(img.getImageView(), vec2(0, 0), vec2(7, 7), (unsigned char)3);
    for (int i = 0; i < 8; ++i) EXPECT_EQ(img(i, i), 3);
    EXPECT_EQ(img(0, 1), 0);

    img.makeZero();
    ImageDraw::drawCircle(img.getImageView(), vec2(4, 4), 1.f, (unsigned char)4);
    EXPECT_EQ(img(4, 4), 4);
    EXPECT_EQ(img(3, 4), 4);
    EXPECT_EQ(img(4, 5), 4);
    EXPECT_EQ(img(3, 3), 0);

    img.makeZero();
    ImageDraw::drawCircleBresenham(img.getImageView(), vec2(4, 4), 3, (unsigned char)5);
    EXPECT_EQ(img(4, 7), 5);
    EXPECT_EQ(img(1, 4), 5);
    EXPECT_EQ(img(4, 4), 0);
    // Circles crossing the border are clipped
    ImageDraw::drawCircleBresenham(img.getImageView(), vec2(0, 0), 20, (unsigned char)5);
}

TEST(ImageFormat, TypeTraits)
{
    for (int t = 0; t < int(TYPE_UNKNOWN); ++t)
    {
        ImageType type = ImageType(t);
        EXPECT_EQ(getType(channels(type), elementType(type)), type);
        EXPECT_EQ(elementSize(type), channels(type) * bitsPerChannel(type) / 8);
        EXPECT_EQ(bitsPerPixel(type), elementSize(type) * 8);
    }
    EXPECT_EQ(elementSize(UC3), 3);
    EXPECT_EQ(elementSize(US4), 8);
    EXPECT_EQ(elementSize(D2), 16);
    EXPECT_EQ(ImageTypeTemplate<ucvec4>::type, UC4);
    EXPECT_EQ(ImageTypeTemplate<usvec3>::type, US3);
    EXPECT_EQ(ImageTypeTemplate<float>::type, F1);
}

TEST(ManagedImage, CreatePitchAndMakeZero)
{
    Image img(3, 5, UC3);
    EXPECT_TRUE(img.valid());
    EXPECT_EQ(img.pitchBytes, 16);  // 15 bytes aligned to 4
    EXPECT_EQ(img.size(), 48);
    img.makeZero();
    img.at<ucvec3>(2, 4) = ucvec3(1, 2, 3);

    Image copy(img.getImageView<ucvec3>());
    EXPECT_EQ(copy.getConstImageView<ucvec3>(), img.getConstImageView<ucvec3>());

    img.create(2, 2, 12, UC1);
    EXPECT_EQ(img.pitchBytes, 12);
    EXPECT_EQ(img.size(), 24);

    img.free();
    EXPECT_FALSE(img.valid());
    img.clear();
    EXPECT_EQ(img.type, TYPE_UNKNOWN);
}

TEST(ManagedImage, Texture)
{
    TemplatedImage<ucvec4> img(2, 2);
    img.getImageView().set(ucvec4(255, 0, 255, 0));
    vec4 c = img.texture(vec2(0.5f, 0.5f));
    EXPECT_FLOAT_EQ(c(0), 1);
    EXPECT_FLOAT_EQ(c(1), 0);
    EXPECT_FLOAT_EQ(c(3), 0);
}

TEST(ImageView, GradientsOfSingleRowOrColumn)
{
    // For width or height 1 the border differences read the pixel at index 1, which is outside of the view.
    // The views below are sub views of a larger image, so that pixel exists and has a different value.
    TemplatedImage<float> img(3, 3);
    for (int y = 0; y < 3; ++y)
        for (int x = 0; x < 3; ++x) img(y, x) = 10 * y + x;

    TemplatedImage<float> g(3, 3);
    g.getImageView().set(-1);

    auto column = img.getImageView().subImageView(0, 1, 3, 1);
    auto gcol   = g.getImageView().subImageView(0, 1, 3, 1);
    column.gx(gcol);
    for (int y = 0; y < 3; ++y) EXPECT_EQ(gcol(y, 0), 0) << y;
    column.gy(gcol);
    EXPECT_EQ(gcol(0, 0), 10);
    EXPECT_EQ(gcol(1, 0), 10);
    EXPECT_EQ(gcol(2, 0), 10);

    auto row  = img.getImageView().subImageView(1, 0, 1, 3);
    auto grow = g.getImageView().subImageView(1, 0, 1, 3);
    row.gy(grow);
    for (int x = 0; x < 3; ++x) EXPECT_EQ(grow(0, x), 0) << x;
    row.gx(grow);
    for (int x = 0; x < 3; ++x) EXPECT_EQ(grow(0, x), 1) << x;

    // Pixels outside of the views are not written.
    EXPECT_EQ(g(0, 0), -1);
    EXPECT_EQ(g(2, 2), -1);
}

TEST(ImageTransformation, DepthToRGBAEmptyRange)
{
    // minD == maxD divided by zero (0/0 = NaN for depths equal to minD, converted to uint8 is undefined).
    TemplatedImage<uint16_t> depth(1, 3);
    depth(0, 0) = 50;
    depth(0, 1) = 100;
    depth(0, 2) = 150;
    TemplatedImage<ucvec4> rgba(1, 3);
    ImageTransformation::depthToRGBA(depth.getConstImageView(), rgba.getImageView(), uint16_t(100), uint16_t(100));
    EXPECT_EQ(rgba(0, 0), ucvec4(0, 0, 0, 255));
    EXPECT_EQ(rgba(0, 1), ucvec4(0, 0, 0, 255));
    EXPECT_EQ(rgba(0, 2), ucvec4(255, 255, 255, 255));

    TemplatedImage<float> fdepth(1, 3);
    fdepth(0, 0) = 0.5f;
    fdepth(0, 1) = 1;
    fdepth(0, 2) = 2;
    ImageTransformation::depthToRGBA(fdepth.getConstImageView(), rgba.getImageView(), 1, 1);
    EXPECT_EQ(rgba(0, 0), ucvec4(0, 0, 0, 255));
    EXPECT_EQ(rgba(0, 1), ucvec4(0, 0, 0, 255));
    EXPECT_EQ(rgba(0, 2), ucvec4(255, 255, 255, 255));

    TemplatedImage<ucvec4> hsv1(1, 1);
    ImageTransformation::depthToRGBA_HSV(fdepth.getConstImageView(), rgba.getImageView(), 1, 1);
    ImageTransformation::depthToRGBA_HSV(fdepth.getConstImageView().subImageView(0, 2, 1, 1), hsv1.getImageView(), 0,
                                         1);
    // depth <= minD maps to 0 and depth > minD to 1 (like a depth of 2 in [0,1])
    EXPECT_EQ(rgba(0, 0), rgba(0, 1));
    EXPECT_EQ(rgba(0, 2), hsv1(0, 0));
    EXPECT_NE(rgba(0, 1), rgba(0, 2));
}
