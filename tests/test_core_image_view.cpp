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
