/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/Core.h"
#include "saiga/core/image/ImageDraw.h"
using namespace Saiga;



float halton(int index, int base)
{
    // Source Wikipedia Halton Sequence
    // https://en.wikipedia.org/wiki/Halton_sequence
    float f = 1;
    float r = 0;
    while (index > 0)
    {
        f     = f / base;
        r     = r + f * (index % base);
        index = index / base;
    }
    return r;
}



float additiveRecurrence(int index, float alpha)
{
    float a = index * alpha;
    return fract(a);
}


int main(int argc, char* args[])
{
    Random::setSeed(93460346346);

    int iw = 512;
    int ih = 512;
    TemplatedImage<unsigned char> img(iw, ih);


    auto set = [&](vec2 p) {
        // assuming p is in [0,1]
        //        p = p * vec2(iw, ih);
        p = p.array() * make_vec2(iw, ih).array();
        ImageDraw::drawCircle(img.getImageView(), p, 4, 255);
    };


    {
        // Random sampling
        img.getImageView().set(128);
        for (int i = 0; i < 100; ++i)
        {
            set(vec2(Random::sampleDouble(0, 1), Random::sampleDouble(0, 1)));
        }
        img.save("random_random.png");
    }


    {
        // Halton sequence
        img.getImageView().set(128);
        for (int i = 0; i < 17; ++i)
        {
            std::cout << vec2(halton(i, 2), halton(i, 3)) * 1.6f << std::endl;
            set(vec2(halton(i, 2), halton(i, 3)));
        }
        img.save("random_halton.png");
    }

    {
        // Additive recurrence
        img.getImageView().set(128);
        float a1 = fract(sqrt(2));
        float a2 = fract(sqrt(11));
        for (int i = 0; i < 100; ++i)
        {
            set(vec2(additiveRecurrence(i, a1), additiveRecurrence(i, a2)));
        }
        img.save("random_ar.png");
    }

    return 0;
}
