/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/Core.h"
#include "saiga/core/image/ImageDraw.h"
#include "saiga/core/math/random.h"
using namespace Saiga;

void test16BitLoadStore()
{
    // Create a gradient image
    int w = 512;
    int h = 512;

    TemplatedImage<unsigned short> img(h, w);

    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            img(i, j) = float(j) / w * ((1 << 16) - 1);
        }
    }

    img.save("test16.png");

    img.load("test16.png");
    img.save("test162.png");
}

void testScaleLinear()
{
    TemplatedImage<ucvec3> img("redie.png");
    {
        // Scale down
        float scale = 0.33f;
        TemplatedImage<ucvec3> img_small(img.h * scale, img.w * scale);
        img.getImageView().copyScaleLinear(img_small.getImageView());
        img_small.save("redie_small.png");
    }

    {
        // Scale up
        float scale = 2.5f;
        TemplatedImage<ucvec3> img_small(img.h * scale, img.w * scale);
        img.getImageView().copyScaleLinear(img_small.getImageView());
        img_small.save("redie_big.png");
    }
}

void testDraw()
{
    int w = 512;
    int h = 512;


    TemplatedImage<ucvec3> img(h, w);
    auto iv = img.getImageView();


    iv.set(ucvec3(0, 0, 0));

    std::vector<vec2> points;
    for (int i = 0; i < 100; ++i)
    {
        points.push_back(linearRand(vec2(0, 0), vec2(h, w)));
    }

    // draw random line between points
    for (int i = 0; i < 100; ++i)
    {
        auto a = Random::uniformInt(0, points.size() - 1);
        auto b = Random::uniformInt(0, points.size() - 1);
        ImageDraw::drawLineBresenham(iv, points[a], points[b], ucvec3(255, 0, 0));
    }

    for (auto p : points)
    {
        if (Random::sampleDouble(0, 1) < 0.5)
            ImageDraw::drawCircle(iv, p, 5, ucvec3(0, 255, 0));
        else
            ImageDraw::drawCircleBresenham(iv, p, 5, ucvec3(0, 0, 255));
    }


    img.save("testDraw.png");
}

int main(int argc, char* argv[])
{
    initSaigaSampleNoWindow();


    testDraw();
    //    test16BitLoadStore();

    testScaleLinear();

    return 0;

    {
        // Test:
        // Read, modify, write a png image.
        TemplatedImage<ucvec3> img("textures/redie.png");
        SAIGA_ASSERT(img.type == UC3);
        ImageView<ucvec3> vimg = img.getImageView();
        vimg.setChannel(0, 0);
        vimg.setChannel(1, 0);

        auto img2 = img;
        img2.save("debug/blue.png");

        // Create a grayscale image from the blue channel
        Image imggray(img.height, img.width, UC1);
        for (int i = 0; i < img.height; ++i)
        {
            for (int j = 0; j < img.width; ++j)
            {
                imggray.at<unsigned char>(i, j) = img.at<ucvec3>(i, j)[2];
            }
        }
        imggray.save("debug/blue_gray.png");
    }

#ifdef SAIGA_USE_FREEIMAGE
    {
        // Test:
        // Read, modify, write a jpg image.
        Image img("textures/redie.jpg");
        ImageView<ucvec3> vimg = img.getImageView<ucvec3>();
        vimg.setChannel(0, 0);
        vimg.setChannel(1, 0);
        img.save("debug/blue.jpg");
    }
#endif

    {
        // Raw format
        TemplatedImage<ucvec3> img("textures/redie.png");

        img.save("debug/raw_test.saigai");

        TemplatedImage<ucvec3> img2("debug/raw_test.saigai");
        img2.save("debug/raw_test.png");
    }
}
