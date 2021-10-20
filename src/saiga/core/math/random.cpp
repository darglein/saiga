/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/math/random.h"

#include "saiga/core/util/assert.h"

#include <chrono>
#include <numeric>
#include <random>

namespace Saiga
{
namespace Random
{
std::mt19937& generator()
{
    static thread_local std::mt19937 gen(std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()));
    return gen;
}


void setSeed(uint64_t seed)
{
    generator().seed(seed);
}



bool sampleBool(double s)
{
    // we need this because the line below is 'inclusive'
    if (s == 1) return true;
    return sampleDouble(0, 1) < s;
}

double sampleDouble(double min, double max)
{
    std::uniform_real_distribution<double> dis(min, max);
    return dis(generator());
}

std::vector<double> StratifiedSample(double min, double max, int count)
{
    std::vector<double> result;
    result.reserve(count);

    double step = (max - min) / count;
    std::uniform_real_distribution<double> dis(0, step);

    for (int i = 0; i < count; ++i)
    {
        double v = min + i * step + dis(generator());
        result.push_back(v);
    }
    return result;
}

int rand()
{
    std::uniform_int_distribution<int> dis(0, std::numeric_limits<int>::max());
    return dis(generator());
}

uint64_t urand64()
{
    std::uniform_int_distribution<uint64_t> dis(0, std::numeric_limits<uint64_t>::max());
    return dis(generator());
}


int uniformInt(int low, int high)
{
    std::uniform_int_distribution<int> dis(low, high);
    return dis(generator());
}

double gaussRand(double mean, double stddev)
{
    std::normal_distribution<double> dis(mean, stddev);
    return dis(generator());
}



std::vector<int> uniqueIndices(int sampleCount, int indexSize)
{
    SAIGA_ASSERT(sampleCount <= indexSize);

    std::vector<bool> used(indexSize, false);
    std::vector<int> data(sampleCount);

    for (int j = 0; j < sampleCount;)
    {
        int s = uniformInt(0, indexSize - 1);
        if (!used[s])
        {
            data[j] = s;
            used[s] = true;
            j++;
        }
    }
    return data;
}


std::vector<int> shuffleSequence(int size)
{
    std::vector<int> indices(size);
    std::iota(indices.begin(), indices.end(), 0);
    std::shuffle(indices.begin(), indices.end(), Random::generator());
    return indices;
}


uint64_t generateTimeBasedSeed()
{
    uint64_t time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

    std::mt19937 gen(time);
    std::uniform_int_distribution<uint64_t> dis(0, std::numeric_limits<uint64_t>::max());
    for (int i = 0; i < 100; ++i)
    {
        time = dis(gen);
    }
    return time;
}


Vec3 ballRand(double radius)
{
    SAIGA_ASSERT(radius >= 0);
    // Credits to random.inl from the glm library
    auto r2 = radius * radius;
    Vec3 low(-radius, -radius, -radius);
    Vec3 high(radius, radius, radius);
    double lenRes;
    Vec3 result;
    do
    {
        result = linearRand(low, high);
        lenRes = result.squaredNorm();
    } while (lenRes > r2);
    return result;
}

Vec3 sphericalRand(double radius)
{
    float z = Random::sampleDouble(-1.0, 1.0);
    float a = Random::sampleDouble(0.0, pi<double>() * 2.0);

    float r = sqrt(1.0 - z * z);

    float x = r * cos(a);
    float y = r * sin(a);

    return Vec3(x, y, z) * radius;
}

}  // namespace Random

vec2 diskRand(float Radius)
{
    vec2 Result(0, 0);
    float LenRadius = 0;

    do
    {
        Result    = linearRand(make_vec2(-Radius), make_vec2(Radius));
        LenRadius = length(Result);
    } while (LenRadius > Radius);

    return Result;
}



}  // namespace Saiga
