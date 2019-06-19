/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/Core.h"

using namespace Saiga;

template <typename T, typename S>
void drawLineBresenham(ImageView<T> img, vec2 start, vec2 end, S color)
{
    // Source
    // https://rosettacode.org/wiki/Bitmap/Bresenham%27s_line_algorithm#C
    int x0 = round(start[0]);
    int y0 = round(start[1]);
    int x1 = round(end[0]);
    int y1 = round(end[1]);

    int dx = abs(x1 - x0), sx = x0 < x1 ? 1 : -1;
    int dy = abs(y1 - y0), sy = y0 < y1 ? 1 : -1;
    int err = (dx > dy ? dx : -dy) / 2, e2;

    for (;;)
    {
        img.clampedWrite(y0, x0, color);
        if (x0 == x1 && y0 == y1) break;
        e2 = err;
        if (e2 > -dx)
        {
            err -= dy;
            x0 += sx;
        }
        if (e2 < dy)
        {
            err += dx;
            y0 += sy;
        }
    }
}



template <typename T, typename S>
void drawCircle(ImageView<T> img, vec2 position, float radius, S color)
{
    for (int dy = -radius; dy <= radius; ++dy)
    {
        for (int dx = -radius; dx <= radius; ++dx)
        {
            float distance = length(vec2(dx, dy));
            if (distance > radius) continue;
            int px = position[0] + dx;
            int py = position[1] + dy;
            img.clampedWrite(py, px, color);
        }
    }
}


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
        p = ele_mult(p, make_vec2(iw, ih));
        drawCircle(img.getImageView(), p, 4, 255);
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
