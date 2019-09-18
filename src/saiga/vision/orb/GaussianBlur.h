#ifndef SAIGA_ORB_GAUSSIANBLUR_H
#define SAIGA_ORB_GAUSSIANBLUR_H

#include "Types.h"

#include <opencv2/imgproc/imgproc.hpp>

namespace Saiga
{
void MakeKernel1D(std::vector<double>& k, double sigma, int sz)
{
    // k = Saiga::gaussianBlurKernel(sz/2, sigma);
    // return;
    double sigmaX  = sigma > 0 ? sigma : ((sz - 1) * 0.5 - 1) * 0.3 + 0.8;
    double scale2X = -0.5 / (sigmaX * sigmaX);
    double sum     = 0;
    int i          = 0;
    for (i = 0; i < sz; ++i)
    {
        double x = i - (sz - 1) * 0.5;
        double t = std::exp(scale2X * x * x);
        k[i]     = t;
        sum += t;
    }
    sum = 1. / sum;
    for (i = 0; i < sz; ++i)
    {
        k[i] *= sum;
    }
}


template <typename T>
void GaussianBlur(img_t& src, img_t& dst, int xK, int yK, double sigmaX, double sigmaY)
{
    SAIGA_ASSERT(xK % 2 == 1 && yK % 2 == 1, "wrong kernel size");
    int hx = xK >> 1;
    int hy = yK >> 1;
    std::vector<double> kernelX(xK);
    std::vector<double> kernelY(yK);
    MakeKernel1D(kernelX, sigmaX, xK);
    MakeKernel1D(kernelY, sigmaY, yK);

    dst = Saiga::ImageView<T>(src.rows, src.cols, src.pitchBytes, dst.data);

    int i, j, k, m, n;
    uchar *srcPtr = &src(0, 0), *dstPtr = &dst(0, 0);
    double *temPtr, *temPtr2;
    int kOffset, endIdx;

    double tem[src.rows * src.cols];
    double sum[src.cols];
    endIdx = src.cols - hx;
    temPtr = tem;

    for (i = 0; i < src.rows; ++i)
    {
        kOffset = -hx;
        for (j = 0; j < hx; ++j)
        {
            *temPtr = 0;
            for (k = kernelX.size() - 1, m = kOffset; k >= 0; --k, ++m)
            {
                if (m < 0) m = -m;
                *temPtr += *(srcPtr + m) * kernelX[k];
            }
            ++temPtr;
            ++kOffset;
        }
        for (j = hx; j < endIdx; ++j)
        {
            *temPtr = 0;
            for (k = (int)kernelX.size() - 1, m = 0; k >= 0; --k, ++m)
            {
                *temPtr += *(srcPtr + m) * kernelX[k];
            }
            ++srcPtr;
            ++temPtr;
        }

        kOffset = 1;
        for (j = endIdx; j < src.cols; ++j)
        {
            *temPtr = 0;
            for (k = (int)kernelX.size() - 1, m = -hx; k >= 0; --k, ++m)
            {
                if (m > (hx - kOffset)) m = -m;
                *temPtr += *(srcPtr + m) * kernelX[k];
            }
            ++srcPtr;
            ++temPtr;
            ++kOffset;
        }

        srcPtr += hx;
    }
    endIdx = src.rows - hy;
    temPtr = temPtr2 = tem;
    dstPtr           = &dst(0, 0);

    for (i = 0; i < src.cols; ++i)
    {
        sum[i] = 0;
    }

    kOffset = 0;
    for (i = 0; i < hy; ++i)
    {
        for (k = hy + kOffset; k >= 0; --k)
        {
            for (j = 0; j < src.cols; ++j)
            {
                sum[j] += *temPtr * kernelY[k];
                ++temPtr;
            }
        }
        for (n = 0; n < src.cols; ++n)
        {
            *dstPtr = (uchar)((double)fabs(sum[n]) + 0.5f);
            sum[n]  = 0;
            ++dstPtr;
        }

        temPtr = temPtr2;
        ++kOffset;
    }

    for (i = hy; i < endIdx; ++i)
    {
        for (k = (int)kernelY.size() - 1; k >= 0; --k)
        {
            for (j = 0; j < src.cols; ++j)
            {
                sum[j] += *temPtr * kernelY[k];
                ++temPtr;
            }
        }

        for (n = 0; n < src.cols; ++n)
        {
            *dstPtr = (uchar)((double)fabs(sum[n]) + 0.5f);
            sum[n]  = 0;
            ++dstPtr;
        }
        temPtr2 += src.cols;
        temPtr = temPtr2;
    }

    kOffset = 1;
    for (i = endIdx; i < src.rows; ++i)
    {
        for (k = (int)kernelY.size() - 1; k >= kOffset; --k)
        {
            for (j = 0; j < src.cols; ++j)
            {
                sum[j] += *temPtr * kernelY[k];
                ++temPtr;
            }
        }
        for (n = 0; n < src.cols; ++n)
        {
            *dstPtr = (uchar)((double)fabs(sum[n]) + 0.5f);
            sum[n]  = 0;
            ++dstPtr;
        }
        temPtr2 += src.cols;
        temPtr = temPtr2;
        ++kOffset;
    }
}
}  // namespace SaigaORB



#endif  // SAIGA_ORB_GAUSSIANBLUR_H
