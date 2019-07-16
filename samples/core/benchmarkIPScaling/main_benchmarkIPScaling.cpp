/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/Core.h"
#include "saiga/core/time/all.h"
using namespace Saiga;

struct ImageProcessing
{
    ImageProcessing(int w, int h) : rgbImage(h, w), floatImage(h, w), grayImage(h, w) {}

    void testRGB2Gray(int threads)
    {
        omp_set_num_threads(threads);

        auto stats = measureObject(50, [&]() {
#pragma omp parallel for
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



        std::cout << "Threads " << threads << " Time (ms): " << t * 1000 << " Bandwidth: " << gbRW / t << std::endl;
    }

    void testRGB2Gray2(int threads)
    {
        ThreadPool tp(threads - 1);

        auto f = [&](int start, int end) {
            for (int i = start; i < end; ++i)
            {
                for (int j = 0; j < rgbImage.w; ++j)
                {
                    auto v = rgbImage(i, j);
                    vec3 vf(v[0], v[1], v[2]);
                    float gray = dot(rgbToGray, vf);
                    grayImage(i, j) += gray;
                }
            }
        };

        std::vector<std::future<void>> rets(threads);
        auto stats = measureObject(50, [&]() {
            int block = rgbImage.h / threads;
            for (int t = 0; t < threads; ++t)
            {
                int start = t * block;
                int end   = (t + 1) * block;
                if (t == threads - 1) end = rgbImage.h;
                rets[t] = tp.enqueue([&]() { f(start, end); });
            }

            // wait for all threads
            for (auto& r : rets)
            {
                r.wait();
            }
        });

        size_t bytesRW = rgbImage.size() + grayImage.size();
        double gbRW    = bytesRW / (1000.0 * 1000.0 * 1000.0);
        double t       = stats.median / 1000.0;



        std::cout << "Threads " << threads << " Time (ms): " << t * 1000 << " Bandwidth: " << gbRW / t << std::endl;
    }
    const vec3 rgbToGray = vec3(0.299f, 0.587f, 0.114f);
    TemplatedImage<ucvec4> rgbImage;
    TemplatedImage<float> floatImage;
    TemplatedImage<unsigned char> grayImage;
};


int main(int, char**)
{
    //    ImageProcessing ip(3000, 1500);
    ImageProcessing ip(640, 480);
    ip.testRGB2Gray(1);
    ip.testRGB2Gray(2);
    ip.testRGB2Gray(3);
    ip.testRGB2Gray(4);

    ip.testRGB2Gray2(1);
    ip.testRGB2Gray2(2);
    ip.testRGB2Gray2(3);
    ip.testRGB2Gray2(4);
    return 0;
}
