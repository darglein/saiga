/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/math/math.h"
#include "saiga/core/math/random.h"

#include <vector>

namespace Saiga
{
/**
 * Samples a discrete probability distribution with the alias method.
 *
 * Sources:
 * Michael D. Vose - A Linear Algorithm For Generating Random Numbers With a Given Distribution
 * https://web.archive.org/web/20131029203736/http://web.eecs.utk.edu/~vose/Publications/random.pdf
 * http://www.keithschwarz.com/darts-dice-coins/
 * https://en.wikipedia.org/wiki/Alias_method
 */

template <typename real_t = float>
class SAIGA_TEMPLATE DiscreteProbabilityDistribution
{
   public:
    static_assert(std::is_floating_point<real_t>::value, "Only floating point types allowed!");
    // O(n)
    DiscreteProbabilityDistribution(std::vector<real_t> probabilities);
    // O(1)
    int sample();

   private:
    //    std::default_random_engine re;
    int n;
    std::vector<int> alias;
    std::vector<real_t> prob;
};

template <typename real_t>
DiscreteProbabilityDistribution<real_t>::DiscreteProbabilityDistribution(std::vector<real_t> probabilities)
//    : re(std::random_device()())
{
    n = probabilities.size();

    // normalize input
    real_t sum = 0;
    for (int i = 0; i < n; ++i)
    {
        sum += probabilities[i];
    }
    for (int i = 0; i < n; ++i)
    {
        probabilities[i] *= (n / sum);
    }

    alias.resize(n);
    prob.resize(n);
    std::vector<int> small,  // contains probabilities <1
        large;               // contains probabilities >1

    for (int i = 0; i < n; ++i)
    {
        if (probabilities[i] < 1)
            small.push_back(i);
        else
            large.push_back(i);
    }

    // Large might be emptied first
    while (!small.empty() && !large.empty())
    {
        // get and remove last elements from both lists
        int s = small[small.size() - 1];
        int l = large[large.size() - 1];
        small.pop_back();
        large.pop_back();

        // shift (1-s) from l to s.
        prob[s]          = probabilities[s];
        alias[s]         = l;
        probabilities[l] = (probabilities[s] + probabilities[l]) - 1;  // l-(1-s)

        // we removed (1-s) from l, so it can be smaller than 1 now
        if (probabilities[l] < 1)
            small.push_back(l);
        else
            large.push_back(l);
    }

    // if either small or large has numbers left their probabilities must be numerical equal to 1
    for (int i : small)
    {
        prob[i]  = 1;
        alias[i] = i;
    }
    for (int i : large)
    {
        prob[i]  = 1;
        alias[i] = i;
    }
}

template <typename real_t>
int DiscreteProbabilityDistribution<real_t>::sample()
{
    int i = Random::uniformInt(0, n - 1);
    return (Random::sampleBool(prob[i])) ? i : alias[i];
}

}  // namespace Saiga
