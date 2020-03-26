/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/core/Core.h"
#include "saiga/core/image/ImageDraw.h"
#include "saiga/core/image/freeimage.h"
#include "saiga/core/image/png_wrapper.h"
#include "saiga/core/math/random.h"

#include "internal/stb_image_read_wrapper.h"
#include "internal/stb_image_write_wrapper.h"

#include "gtest/gtest.h"

#include <filesystem>
using namespace Saiga;


template <typename T>
TemplatedImage<T> randomImage(int h, int w)
{
    TemplatedImage<T> img(h, w);
    for (int i = 0; i < img.size(); ++i)
    {
        img.data8()[i] = (uint8_t)Saiga::Random::uniformInt(0, 255);
    }
    return img;
}

// Creates a templated image of type T and saves it on a disk.
// Then it loads the file again and checks if the content is the same.
template <typename T>
void testSaveLoadLibPNG(const TemplatedImage<T>& img)
{
    std::string file = "loadstoretest_libpng.png";
    std::filesystem::remove(file);
    EXPECT_TRUE(PNG::save(img, file));

    TemplatedImage<T> img2;
    EXPECT_TRUE(PNG::load(img2, file));

    EXPECT_EQ(img.dimensions(), img2.dimensions());
    EXPECT_EQ(img, img2);
}



// Creates a templated image of type T and saves it on a disk.
// Then it loads the file again and checks if the content is the same.
template <typename T>
void testSaveLoadFreeimage(const TemplatedImage<T>& img, const std::string& type = "png")
{
    std::string file = "loadstoretest_freeimage." + type;
    std::filesystem::remove(file);
    EXPECT_TRUE(FIP::save(file, img));

    TemplatedImage<T> img2;
    img2.makeZero();

    EXPECT_TRUE(FIP::load(file, img2, 0));

    EXPECT_EQ(img.dimensions(), img2.dimensions());
    EXPECT_EQ(img.getConstImageView(), img2.getConstImageView());
}


TEST(ImageLoadStore, UC1)
{
    using T  = unsigned char;
    auto img = randomImage<T>(128, 128);
    testSaveLoadLibPNG(img);
    testSaveLoadFreeimage(img);
}

TEST(ImageLoadStore, UC2)
{
    using T  = ucvec2;
    auto img = randomImage<T>(128, 128);
    testSaveLoadLibPNG(img);

    // not supportet
    //    testSaveLoadFreeimage(img);
}

TEST(ImageLoadStore, UC3)
{
    using T  = ucvec3;
    auto img = randomImage<T>(128, 128);
    testSaveLoadLibPNG(img);
    testSaveLoadFreeimage(img, "png");
    //    testSaveLoadFreeimage(img, "jpg");
    testSaveLoadFreeimage(img, "bmp");
    //    testSaveLoadFreeimage(img, "tiff");
}

TEST(ImageLoadStore, UC4)
{
    using T  = ucvec4;
    auto img = randomImage<T>(128, 128);
    testSaveLoadLibPNG(img);
    testSaveLoadFreeimage(img, "png");
    //    testSaveLoadFreeimage(img, "bmp");
    //    testSaveLoadFreeimage(img, "jpg");
    //    testSaveLoadFreeimage(img, "tiff");
}

TEST(ImageLoadStore, US1)
{
    using T  = unsigned short;
    auto img = randomImage<T>(128, 128);
    testSaveLoadLibPNG(img);
    testSaveLoadFreeimage(img);
}

TEST(ImageLoadStore, US2)
{
    using T  = usvec2;
    auto img = randomImage<T>(128, 128);
    testSaveLoadLibPNG(img);

    // not supportet
    // testSaveLoadFreeimage(img);
}

TEST(ImageLoadStore, US3)
{
    using T  = usvec3;
    auto img = randomImage<T>(128, 128);
    testSaveLoadLibPNG(img);
    testSaveLoadFreeimage(img);
}

TEST(ImageLoadStore, US4)
{
    using T  = usvec4;
    auto img = randomImage<T>(128, 128);
    testSaveLoadLibPNG(img);
    testSaveLoadFreeimage(img);
}

TEST(ImageLoadStore, I1)
{
    using T  = int;
    auto img = randomImage<T>(128, 128);
    //    testSaveLoadFreeimage(img, "png");
}

TEST(ImageLoadStore, UI1)
{
    using T  = unsigned int;
    auto img = randomImage<T>(128, 128);
    //    testSaveLoadFreeimage(img, "png");
}


TEST(ImageLoadStore, F1)
{
    using T  = float;
    auto img = randomImage<T>(128, 128);
    //    testSaveLoadFreeimage(img, "tiff");
}
