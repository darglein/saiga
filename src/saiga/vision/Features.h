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
template<typename T = float>
struct KeyPoint
{
    using Vec2 = Eigen::Matrix<T, 2, 1>;

    Vec2 point; // Points coordinates (x,y) in image space
    T size;
    T angle;
    T response;
    int octave;
};

// Some common feature descriptors
using DescriptorORB = std::array<int32_t,8>;
using DescriptorSIFT = std::array<float,128>;


#if 1
// use the popcnt instruction
// this will be the fastest implementation if it is available
// more here: https://github.com/kimwalisch/libpopcnt
inline uint32_t popcnt32(uint32_t x)
{
  __asm__ ("popcnt %1, %0" : "=r" (x) : "0" (x));
  return x;
}
#else
inline uint32_t popcnt32(uint32_t v)
{
    v              = v - ((v >> 1) & 0x55555555);
    v              = (v & 0x33333333) + ((v >> 2) & 0x33333333);
    return (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
}
#endif

// Compute the hamming distance between the two descriptors
// Same implementation as ORB SLAM
// http://graphics.stanford.edu/~seander/bithacks.html#CountBitsSetParallel
inline int distance(const DescriptorORB& a,const DescriptorORB& b)
{
    auto pa = a.data();
    auto pb = b.data();
    int dist = 0;
    for (int i = 0; i < 8; i++, pa++, pb++)
    {
        uint32_t v = *pa ^ *pb;

        // TODO: if this is really a bottleneck we can also use AVX-2
        // to gain around 25% more performance
        // according to this source:
        // https://github.com/kimwalisch/libpopcnt
        dist += popcnt32(v);
//        v              = v - ((v >> 1) & 0x55555555);
//        v              = (v & 0x33333333) + ((v >> 2) & 0x33333333);
//        dist += (((v + (v >> 4)) & 0xF0F0F0F) * 0x1010101) >> 24;
    }

    return dist;
}



// Compute the euclidean distance between the descriptors
inline int distance(const DescriptorSIFT& a,const DescriptorSIFT& b)
{
    float sumSqr =0;
    for (int i = 0; i < 128; ++i)
    {
        auto c = a[i] - b[i];
        sumSqr += c * c;
    }
    return sqrt(sumSqr);
}

}  // namespace Saiga
