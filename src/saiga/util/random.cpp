/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/util/random.h"

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

}  // namespace Random
}  // namespace Saiga
