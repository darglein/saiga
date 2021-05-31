/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/assert.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <vector>

namespace Saiga
{
template <typename T = double>
class Statistics
{
   public:
    Statistics() {}
    Statistics(const std::vector<T>& data);

    int numValues = 0;
    T min         = 0;
    T max         = 0;
    T median      = 0;
    T mean = 0, sum = 0;
    T variance = 0, sdev = 0;
    T rms = 0;
};

template <typename T>
Statistics<T>::Statistics(const std::vector<T>& _data)
{
    std::vector<T> data = _data;
    numValues           = data.size();

    if (numValues == 0) return;

    std::sort(data.begin(), data.end());

    min    = data.front();
    max    = data.back();
    median = data[data.size() / 2];

    rms = 0;
    sum = 0;
    for (auto& d : data) sum += d;
    mean = sum / numValues;

    variance = 0;
    for (auto d : data)
    {
        auto center = d - mean;
        variance += center * center;
        rms += d * d;
    }
    variance /= numValues;

    sdev = std::sqrt(variance);
    rms  = std::sqrt(rms / numValues);
}

template <typename T>
std::ostream& operator<<(std::ostream& stream, const Statistics<T>& object)
{
    stream << "Num         = [" << object.numValues << "]" << std::endl
           << "Min,Max     = [" << object.min << "," << object.max << "]" << std::endl
           << "Mean,Median,Rms = [" << object.mean << "," << object.median << "," << object.rms << "]" << std::endl
           << "sdev,var    = [" << object.sdev << "," << object.variance << "]";
    return stream;
}

template <typename T>
Vector<T, -1> gaussianBlurKernel1d(int radius, T sigma)
{
    const int ELEMENTS = radius * 2 + 1;
    Vector<T, -1> kernel(ELEMENTS);
    T ivar2 = 1.0f / (2.0f * sigma * sigma);
    for (int x = -radius; x <= radius; x++)
    {
        T d2               = x * x;
        kernel[x + radius] = std::exp(-d2 * ivar2);
    }
    // normalize
    T s = kernel.array().sum();
    return kernel / s;
}


template <typename T>
Matrix<T, -1, -1> gaussianBlurKernel2d(int radius, T sigma)
{
    const int ELEMENTS = radius * 2 + 1;
    Matrix<T, -1, -1> kernel(ELEMENTS, ELEMENTS);
    T ivar2 = 1.0f / (2.0f * sigma * sigma);
    for (int y = -radius; y <= radius; y++)
    {
        for (int x = -radius; x <= radius; x++)
        {
            float d2                       = x * x + y * y;
            kernel(y + radius, x + radius) = std::exp(-d2 * ivar2);
        }
    }
    // normalize
    T s = kernel.array().sum();
    return kernel / s;
}



template <typename T>
void applyFilter1D(const std::vector<T>& src, std::vector<T>& dst, const std::vector<T>& kernel)
{
    SAIGA_ASSERT(src.size() == dst.size());
    SAIGA_ASSERT(kernel.size() % 2 == 1);

    int radius = kernel.size() / 2;

    for (int x = 0; x < dst.size(); ++x)
    {
        T sum(0);
        for (int i = -radius; i <= radius; i++)
        {
            int id = i + x;
            id     = std::min<int>(src.size() - 1, std::max(id, 0));
            sum += src[id] * kernel[i + radius];
        }
        dst[x] = sum;
    }
}

}  // namespace Saiga
