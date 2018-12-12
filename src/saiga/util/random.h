/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/util/math.h"



namespace Saiga
{
/**
 * Simple Random numbers that are created by c++11 random engines.
 * These function use static thread local generators.
 * -> They are created on the first use
 * -> Can be used in multi threaded programs
 */
namespace Random
{
/**
 * Returns true with a probability of 's'.
 * s must be in the range [0,1].
 */
SAIGA_GLOBAL bool sampleBool(double s);

/**
 * Returns a uniform random value in the given range.
 */
SAIGA_GLOBAL double sampleDouble(double min, double max);

/**
 * Uniform integer in this range.
 * Note:
 * The high-bound is inclusive!!!
 */
SAIGA_GLOBAL int uniformInt(int low, int high);


/**
 * A normal-distributed random value
 */
SAIGA_GLOBAL double gaussRand(double mean = 0, double stddev = 1);



/**
 * Similar to std::rand but with thread save c++11 generators
 */
SAIGA_GLOBAL int rand();

}  // namespace Random
}  // namespace Saiga
