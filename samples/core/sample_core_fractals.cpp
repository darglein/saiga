/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"

#ifndef SAIGA_FULL_EIGEN
#    include "saiga/colorize.h"
#endif

#include "saiga/core/Core.h"

#include <complex>

using namespace Saiga;



template <typename T>
struct ClassicMandelbrot
{
   public:
    ClassicMandelbrot(std::complex<T> c) : c(c) {}
    std::complex<T> operator()(const std::complex<T>& x) { return x * x + c; }

   private:
    std::complex<T> c;
};



template <typename T>
struct Test
{
   public:
    Test(std::complex<T> c) : c(c) {}
    std::complex<T> operator()(const std::complex<T>& z)
    {
        auto za = 1.0 - z * z * z / 6.0;
        auto zn = z - z * z / 2.0;

        return za / (zn * zn) + c;
    }

   private:
    std::complex<T> c;
};



int main(int argc, char* args[])
{
    initSaigaSampleNoWindow();

    using T      = double;
    using Number = std::complex<T>;

    // =========================================================

    // Image size
    int w = 1000;
    int h = 1000;

    // Fractal bounds
    //    Number center(-.74364990, .13188204);
    //    double diameter = .00073801;
    Number center(0.1, 0.4);
    double diameter = 0.1;

    // =========================================================



    TemplatedImage<ucvec3> img(w, h);
    {
        SAIGA_BLOCK_TIMER();
#pragma omp parallel for
        for (int i = 0; i < h; ++i)
        {
            for (int j = 0; j < w; ++j)
            {
                int iteration     = 0;
                int max_iteration = 1000;


#if 0

                Number c((T(j) / w), (T((h - i - 1)) / h));
                c = (c - Number(0.5, 0.5)) * diameter + center;
                Number z(0, 0);
                ClassicMandelbrot<T> f(c);
                while (z.imag() * z.imag() + z.real() * z.real() <= 4 && iteration < max_iteration)
                {
                    z         = f(z);
                    iteration = iteration + 1;
                }

#else

                Number z((T(j) / w), (T((h - i - 1)) / h));
                z = (z - Number(0.5, 0.5)) * diameter + center;
                Number c(-0.4, 0.6);

                ClassicMandelbrot<T> f(c);
                //                Test<T> f(c);
                while (z.imag() * z.imag() + z.real() * z.real() <= 4 && iteration < max_iteration)
                {
                    //                      z = z * z + c;
                    z = f(z);

                    iteration = iteration + 1;
                }
#endif

#ifndef SAIGA_FULL_EIGEN
                float alpha = saturate(double(iteration) / max_iteration * 4);

                if (iteration == max_iteration) alpha = 0;

                vec3 color = clamp(colorizePlasma(alpha), vec3(0), vec3(1));

                //                vec3 color = clamp(vec3(alpha ),vec3(0),vec3(1));

                color *= 255.f;
                img(i, j) = ucvec3(color[0], color[1], color[2]);
#endif
            }
        }
    }
    img.save("fractals.png");
}
