/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"

#ifndef WIN32
#    include "saiga/colorize.h"
#endif

#include "saiga/core/Core"

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


int main(int argc, char* args[])
{
    Saiga::SaigaParameters saigaParameters;
    Saiga::initSample(saigaParameters);
    Saiga::initSaiga(saigaParameters);

    using T      = double;
    using Number = std::complex<T>;

    // =========================================================

    // Image size
    int w = 1500;
    int h = 1500;

    // Fractal bounds
    Number center(-.74364990, .13188204);
    double diameter = .00073801;

    // =========================================================



    TemplatedImage<ucvec3> img(w, h);
    {
        SAIGA_BLOCK_TIMER();
#pragma omp parallel for
        for (int i = 0; i < h; ++i)
        {
            for (int j = 0; j < w; ++j)
            {
                Number c((T(j) / w), (T((h - i - 1)) / h));
                c = (c - Number(0.5, 0.5)) * diameter + center;

                Number z(0, 0);

                ClassicMandelbrot<T> f(c);
                int iteration     = 0;
                int max_iteration = 1000;

                while (z.imag() * z.imag() + z.real() * z.real() <= 4 && iteration < max_iteration)
                {
                    z         = f(z);
                    iteration = iteration + 1;
                }
                float alpha = iteration;

#ifndef WIN32
                vec3 color = colorizeMagma(alpha / 500) * 255.f;
                img(i, j)  = ucvec3(color[0], color[1], color[2]);
#endif
            }
        }
    }
    img.save("fractals.png");
}
