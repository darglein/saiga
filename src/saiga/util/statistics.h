/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include <saiga/util/assert.h>
#include <algorithm>
#include <cmath>
#include <vector>

namespace Saiga {

template<typename T>
class Statistics{
public:
    Statistics(const std::vector<T>& data);

    int numValues;
    T min;
    T max;
    T median;
    T mean, sum;
    T variance, sdev;
};

template<typename T>
Statistics<T>::Statistics(const std::vector<T>& _data)
{
    std::vector<T> data = _data;
    numValues = data.size();

    if(numValues  == 0)
        return;

    std::sort(data.begin(),data.end());

    min = data.front();
    max = data.back();
    median = data[data.size()/2];

    sum = 0;
    for(auto &d : data)
        sum += d;
    mean = sum / numValues;

    variance = 0;
    for(auto d : data){
        d = d - mean;
        variance += d * d;
    }
    variance /= numValues;

    sdev = std::sqrt(variance);
}

template<typename T>
std::ostream &operator<<(std::ostream &stream, const Statistics<T> &object)
{
    stream << "Num         = [" << object.numValues << "]" << std::endl <<
              "Min,Max     = [" << object.min << "," << object.max << "]" << std::endl <<
              "Mean,Median = [" << object.mean << "," << object.median << "]" << std::endl <<
              "sdev,var    = [" << object.sdev << "," << object.variance << "]";
    return stream;
}

template<typename T>
std::vector<T>  gaussianBlurKernel(int radius, T sigma)
{
    const int ELEMENTS = radius * 2 + 1;
    std::vector<T> kernel(ELEMENTS);
    T ivar2 = 1.0f / (2.0f * sigma * sigma);
    T kernelSum(0);
    for (int j=-radius; j<=radius; j++)
    {
        kernel[j+radius] = std::exp(-j*j*ivar2);
        kernelSum += kernel[j+radius];
    }
    //normalize
    for (int j=-radius;j<=radius;j++)
    {
        kernel[j+radius] /= kernelSum;
    }
    return kernel;
}

template<typename T>
void applyFilter1D(const std::vector<T>& src, std::vector<T>& dst, const std::vector<T>& kernel)
{
    SAIGA_ASSERT(src.size() == dst.size());
    SAIGA_ASSERT(kernel.size() % 2 == 1);

    int radius = kernel.size() / 2;

    for(int x = 0; x < dst.size(); ++x)
    {
    T sum(0);
    for (int i=-radius;i<=radius;i++){
        int id = i + x;
        id = std::min<int>(src.size()-1,std::max(id,0));
        sum += src[id] * kernel[i+radius];
    }
        dst[x] = sum;
    }
}

}
