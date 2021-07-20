/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/Core.h"
#include "saiga/core/time/all.h"
#include "saiga/core/util/Thread/omp.h"
using namespace Saiga;

struct ImageProcessing
{
    ImageProcessing(int w, int h) : rgbImage(h, w), floatImage(h, w), grayImage(h, w) {}

    double testRGB2Gray(int threads)
    {
        //        omp_set_num_threads(threads);

        auto stats = measureObject(100, [&]() {
#pragma omp parallel for num_threads(threads)
            for (int i = 0; i < rgbImage.h; ++i)
            {
                for (int j = 0; j < rgbImage.w; ++j)
                {
                    auto v = rgbImage(i, j);
                    vec3 vf(v[0], v[1], v[2]);
                    float gray = dot(rgbToGray, vf);
                    grayImage(i, j) += gray;
                }
            }
        });

        size_t bytesRW = rgbImage.size() + grayImage.size();
        double gbRW    = bytesRW / (1000.0 * 1000.0 * 1000.0);
        double t       = stats.median / 1000.0;
        double bw      = gbRW / t;


        std::cout << "Threads " << threads << " Time (ms): " << t * 1000 << " Bandwidth: " << bw << std::endl;
        return bw;
    }


    const vec3 rgbToGray = vec3(0.299f, 0.587f, 0.114f);
    TemplatedImage<ucvec4> rgbImage;
    TemplatedImage<float> floatImage;
    TemplatedImage<unsigned char> grayImage;
};


int main(int, char**)
{
    ImageProcessing ip(640 * 4, 480 * 4);

    int maxThreads = OMP::getMaxThreads();
#pragma omp parallel
    maxThreads = OMP::getNumThreads();

    // just add 2 more so we see if the performance actually falls of now
    maxThreads += 2;

    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    std::vector<double> bws;
    for (int i = 1; i <= maxThreads; ++i)
    {
        bws.push_back(ip.testRGB2Gray(i));
    }
    auto lvl1 = bws.front();

    for (int i = 1; i <= maxThreads; ++i)
    {
        std::cout << "Scaling " << i << " -> " << bws[i - 1] / lvl1 << std::endl;
    }


    return 0;
}
