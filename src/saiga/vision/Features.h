/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionIncludes.h"


namespace Saiga
{
/**
 * Identical to OpenCV's Keypoint struct except it uses
 * Eigen Points and no 'class_id' member.
 * @brief The Keypoint struct
 */
template <typename T = float>
struct KeyPoint
{
    using Vec2 = Eigen::Matrix<T, 2, 1>;

    Vec2 point;  // Points coordinates (x,y) in image space
    T size;
    T angle;
    T response;
    int octave;

    bool operator==(const KeyPoint& other) const
    {
        return point == other.point && size == other.size && angle == other.angle && response == other.response &&
               octave == other.octave;
    }
};

// Some common feature descriptors
using DescriptorORB  = std::array<uint64_t, 4>;
using DescriptorSIFT = std::array<float, 128>;



#ifndef WIN32
// use the popcnt instruction
// this will be the fastest implementation if it is available
// more here: https://github.com/kimwalisch/libpopcnt
inline uint32_t popcnt(uint32_t x)
{
    __asm__("popcnt %1, %0" : "=r"(x) : "0"(x));
    return x;
}
inline uint64_t popcnt(uint64_t x)
{
    __asm__("popcnt %1, %0" : "=r"(x) : "0"(x));
    return x;
}
#else
// Bit count function got from:
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetKernighan
inline uint32_t popcnt(uint32_t v)
{
    v = v - ((v >> 1) & 0x55555555);
    v = (v & 0x33333333) + ((v >> 2) & 0x33333333);
    return (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
}
inline uint64_t popcnt(uint64_t v)
{
    v = v - ((v >> 1) & (uint64_t) ~(uint64_t)0 / 3);
    v = (v & (uint64_t) ~(uint64_t)0 / 15 * 3) + ((v >> 2) & (uint64_t) ~(uint64_t)0 / 15 * 3);
    v = (v + (v >> 4)) & (uint64_t) ~(uint64_t)0 / 255 * 15;
    return (uint64_t)(v * ((uint64_t) ~(uint64_t)0 / 255)) >> (sizeof(uint64_t) - 1) * CHAR_BIT;
}
#endif

// Compute the hamming distance between the two descriptors
// Same implementation as ORB SLAM
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
inline int distance(const DescriptorORB& a, const DescriptorORB& b)
{
    int dist = 0;
    for (int i = 0; i < (int)a.size(); i++)
    {
        auto v = a[i] ^ b[i];
        // TODO: if this is really a bottleneck we can also use AVX-2
        // to gain around 25% more performance
        // according to this source:
        // https://github.com/kimwalisch/libpopcnt
        dist += popcnt(v);
    }

    return dist;
}



// Compute the euclidean distance between the descriptors
inline float distance(const DescriptorSIFT& a, const DescriptorSIFT& b)
{
    float sumSqr = 0;
    for (int i = 0; i < 128; ++i)
    {
        auto c = a[i] - b[i];
        sumSqr += c * c;
    }
    return sqrt(sumSqr);
}


template <typename T>
struct MeanMatcher
{
    inline int bestDescriptorFromArray(const std::vector<T>& descriptors)
    {
        static_assert(std::is_same<T, DescriptorORB>::value, "Only implemented for ORB so far.");
        if (descriptors.size() == 0) return -1;
        // Compute distances between them
        size_t N = descriptors.size();
        std::vector<std::vector<int>> Distances(N, std::vector<int>(N));

        for (size_t i = 0; i < N; i++)
        {
            Distances[i][i] = 0;
            for (size_t j = i + 1; j < N; j++)
            {
                int distij      = distance(descriptors[i], descriptors[j]);
                Distances[i][j] = distij;
                Distances[j][i] = distij;
            }
        }

        // Take the descriptor with least median distance to the rest
        int BestMedian = INT_MAX;
        int BestIdx    = 0;
        for (size_t i = 0; i < N; i++)
        {
            // vector<int> vDists(Distances[i],Distances[i]+N);
            auto& vDists = Distances[i];
            sort(vDists.begin(), vDists.end());
            int median = vDists[0.5 * (N - 1)];

            if (median < BestMedian)
            {
                BestMedian = median;
                BestIdx    = i;
            }
        }
        return BestIdx;
    }



};


}  // namespace Saiga
