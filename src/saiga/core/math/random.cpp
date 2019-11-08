/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/math/random.h"

#include "saiga/core/util/assert.h"

#include <chrono>
#include <random>

namespace Saiga
{
namespace Random
{
inline std::mt19937& generator()
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



}  // namespace Random

float linearRand(float low, float high)
{
    return Saiga::Random::sampleDouble(low, high);
}

vec2 linearRand(const vec2& low, const vec2& high)
{
    return vec2(Saiga::Random::sampleDouble(low[0], high[0]), Saiga::Random::sampleDouble(low[1], high[1]));
}

vec3 linearRand(const vec3& low, const vec3& high)
{
    return vec3(Saiga::Random::sampleDouble(low[0], high[0]), Saiga::Random::sampleDouble(low[1], high[1]),
                Saiga::Random::sampleDouble(low[2], high[2]));
}

vec4 linearRand(const vec4& low, const vec4& high)
{
    return vec4(Saiga::Random::sampleDouble(low[0], high[0]), Saiga::Random::sampleDouble(low[1], high[1]),
                Saiga::Random::sampleDouble(low[2], high[2]), Saiga::Random::sampleDouble(low[3], high[3]));
}

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

// namespace Random
}  // namespace Saiga
