/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/assert.h"

#include <cmath>
#include <cstdint>

namespace Saiga
{
namespace Engine
{
/**
 * PCG32 random number generator (pcg32 of pcg-random.org: 64-bit LCG state, XSH RR output, the increment selects one
 * of 2^63 streams). The same seed and stream give the same sequence on every machine and compiler: no
 * std::*_distribution is used (their results differ between standard libraries). docs/engine_design.md §7.10, §8.6.
 */
class Rng
{
   public:
    struct State
    {
        uint64_t state     = 0;
        uint64_t increment = 1;
        bool operator==(const State& o) const { return state == o.state && increment == o.increment; }
        bool operator!=(const State& o) const { return !(*this == o); }
    };

    // pcg32_srandom of the reference implementation: Rng(42, 54) gives 0xa15c02b7, 0x7b47f409, ...
    explicit Rng(uint64_t seed = 0, uint64_t stream = 0)
    {
        s.state     = 0;
        s.increment = (stream << 1u) | 1u;
        nextU32();
        s.state += seed;
        nextU32();
    }

    uint32_t nextU32()
    {
        uint64_t old        = s.state;
        s.state             = old * 6364136223846793005ULL + s.increment;
        uint32_t xorshifted = uint32_t(((old >> 18u) ^ old) >> 27u);
        uint32_t rot        = uint32_t(old >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((32u - rot) & 31u));
    }

    uint64_t nextU64()
    {
        uint64_t hi = nextU32();
        return (hi << 32) | nextU32();
    }

    // [0, 1) with 53 random bits (two outputs)
    double uniform()
    {
        uint64_t a = nextU32() >> 5;  // 27 bits
        uint64_t b = nextU32() >> 6;  // 26 bits
        return double((a << 26) | b) * (1.0 / 9007199254740992.0);
    }

    // [a, b) for a < b (a for a == b). std::fma: one rounding on every compiler and CPU (a * b + c may or may not be
    // contracted to an FMA instruction depending on the build, which changes the last bit). The result can round up
    // to b for u close to 1: then the largest double below b is returned.
    double uniform(double a, double b)
    {
        SAIGA_ASSERT(a <= b, "Rng::uniform: a > b");
        double r = std::fma(b - a, uniform(), a);
        return (r < b || a == b) ? r : std::nextafter(b, a);
    }

    // Uniform in [a, b] (both inclusive), without modulo bias. Per attempt one output for spans up to 2^32, two above;
    // a rejected attempt draws again.
    int64_t range(int64_t a, int64_t b)
    {
        SAIGA_ASSERT(a <= b, "Rng::range: a > b");
        // Number of values minus one, in unsigned arithmetic (the full int64 range has 2^64 values)
        uint64_t spanMinusOne = uint64_t(b) - uint64_t(a);
        if (spanMinusOne <= 0xFFFFFFFFull)
        {
            // Lemire's multiply-shift with rejection
            uint64_t span = spanMinusOne + 1;  // <= 2^32
            uint64_t m    = uint64_t(nextU32()) * span;
            uint32_t low  = uint32_t(m);
            if (low < span)
            {
                uint32_t threshold = uint32_t((0x100000000ull - span) % span);
                while (low < threshold)
                {
                    m   = uint64_t(nextU32()) * span;
                    low = uint32_t(m);
                }
            }
            return int64_t(uint64_t(a) + (m >> 32));
        }
        if (spanMinusOne == UINT64_MAX) return int64_t(nextU64());
        // Rejection sampling on 64 bits: accept below the largest multiple of span
        uint64_t span  = spanMinusOne + 1;
        uint64_t limit = UINT64_MAX - (UINT64_MAX % span + 1) % span;
        uint64_t x;
        do
        {
            x = nextU64();
        } while (x > limit);
        return int64_t(uint64_t(a) + x % span);
    }

    State state() const { return s; }
    void setState(const State& state)
    {
        SAIGA_ASSERT(state.increment & 1u, "Rng::setState: the increment must be odd");
        s = state;
    }

   private:
    State s;
};

}  // namespace Engine
}  // namespace Saiga
