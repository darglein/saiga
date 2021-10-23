/*M///////////////////////////////////////////////////////////////////////////////////////
//
//  IMPORTANT: READ BEFORE DOWNLOADING, COPYING, INSTALLING OR USING.
//
//  By downloading, copying, installing or using the software you agree to this license.
//  If you do not agree to this license, do not download, install,
//  copy or use the software.
//
//
//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2000-2008, Intel Corporation, all rights reserved.
// Copyright (C) 2009, Willow Garage Inc., all rights reserved.
// Third party copyrights are property of their respective owners.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors "as is" and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are disclaimed.
// In no event shall the Intel Corporation or contributors be liable for any direct,
// indirect, incidental, special, exemplary, or consequential damages
// (including, but not limited to, procurement of substitute goods or services;
// loss of use, data, or profits; or business interruption) however caused
// and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.
//
//M*/
/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "saiga/cuda/cudaHelper.h"
//
#include "saiga/core/time/all.h"
#include "saiga/cuda/device_helper.h"
#include "saiga/cuda/reduce.h"
#include "saiga/vision/features/Features.h"
#include "saiga/vision/features/OrbPattern.h"

#ifdef SAIGA_VISION

#include "OrbDescriptors.h"
namespace Saiga
{
namespace CUDA
{
const int HALF_PATCH_SIZE = 15;

__constant__ unsigned char c_pattern[sizeof(int2) * 512];

__constant__ int c_u_max[32];

ORB::ORB()
{
    auto pattern = Saiga::ORBPattern::DescriptorPattern();
    static_assert(sizeof(Saiga::ivec2) == 2 * sizeof(int), "laksdf");
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(c_pattern, pattern.data(), sizeof(Saiga::ivec2) * pattern.size()));

    auto u_max = Saiga::ORBPattern::AngleUmax();
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(c_u_max, u_max.data(), u_max.size() * sizeof(int)));
}


__global__ void calcOrb_kernel(cudaTextureObject_t tex, Saiga::ImageView<unsigned char> image,
                               Saiga::ArrayView<Saiga::KeyPoint<float>> keypoints,
                               Saiga::ArrayView<Saiga::DescriptorORB> descriptors)
{
    int id  = blockIdx.x;
    int tid = threadIdx.x;
    if (id >= keypoints.size()) return;

    __shared__ unsigned char result[32];

    const auto& kpt     = keypoints[id];
    float2 loc          = {kpt.point(0), kpt.point(1)};
    const auto* pattern = ((int2*)c_pattern) + 16 * tid;

    unsigned char* desc  = (unsigned char*)&descriptors[id];
    const float factorPI = (float)(pi<float>() / 180.f);
    float angle          = (float)kpt.angle * factorPI;
    float a = (float)cosf(angle), b = (float)sinf(angle);

    int t0, t1, val;

    auto GET_VALUE = [&](int idx) -> int {
        int2 pat = pattern[idx];
        float fx = loc.x + (pat.x * a - pat.y * b);
        float fy = loc.y + (pat.x * b + pat.y * a);
        //        int x    = __float2int_rn(fx);
        //        int y    = __float2int_rn(fy);

        //        image.mirrorToEdge(y, x);
        //        CUDA_ASSERT(image.inImage(y, x));
        //        return image(y, x);
        return tex2D<unsigned char>(tex, fx + 0.5, fy + 0.5);
    };

    t0  = GET_VALUE(0);
    t1  = GET_VALUE(1);
    val = t0 < t1;
    t0  = GET_VALUE(2);
    t1  = GET_VALUE(3);
    val |= (t0 < t1) << 1;
    t0 = GET_VALUE(4);
    t1 = GET_VALUE(5);
    val |= (t0 < t1) << 2;
    t0 = GET_VALUE(6);
    t1 = GET_VALUE(7);
    val |= (t0 < t1) << 3;
    t0 = GET_VALUE(8);
    t1 = GET_VALUE(9);
    val |= (t0 < t1) << 4;
    t0 = GET_VALUE(10);
    t1 = GET_VALUE(11);
    val |= (t0 < t1) << 5;
    t0 = GET_VALUE(12);
    t1 = GET_VALUE(13);
    val |= (t0 < t1) << 6;
    t0 = GET_VALUE(14);
    t1 = GET_VALUE(15);
    val |= (t0 < t1) << 7;


    result[threadIdx.x] = (unsigned char)val;

    if (threadIdx.x < 8)
    {
        auto data_int = (int*)result;

        ((int*)desc)[threadIdx.x] = data_int[threadIdx.x];
    }
}



void ORB::ComputeDescriptors(cudaTextureObject_t tex, Saiga::ImageView<unsigned char> image,
                             Saiga::ArrayView<Saiga::KeyPoint<float>> _keypoints,
                             Saiga::ArrayView<Saiga::DescriptorORB> _descriptors, cudaStream_t stream)
{
    if (_keypoints.empty())
    {
        return;
    }
    SAIGA_ASSERT(_keypoints.size() == _descriptors.size());
    calcOrb_kernel<<<_keypoints.size(), 32, 0, stream>>>(tex, image, _keypoints, _descriptors);
}



__global__ void IC_Angle_kernel(cudaTextureObject_t tex, Saiga::ImageView<unsigned char> image,
                                Saiga::ArrayView<Saiga::KeyPoint<float>> keypoints)
{
    const int ptidx = blockIdx.x * blockDim.y + threadIdx.y;

    if (ptidx >= keypoints.size()) return;


    int m_01 = 0, m_10 = 0;
    const int2 loc = make_int2(keypoints[ptidx].point(0), keypoints[ptidx].point(1));

    // Treat the center line differently, v=0
    for (int u = threadIdx.x - HALF_PATCH_SIZE; u <= HALF_PATCH_SIZE; u += blockDim.x)
    {
        m_10 += u * tex2D<unsigned char>(tex, loc.x + u, loc.y);
    }

    m_10 = Saiga::CUDA::warpReduceSum<int, 32, false>(m_10);

    for (int v = 1; v <= HALF_PATCH_SIZE; ++v)
    {
        // Proceed over the two lines
        int v_sum   = 0;
        int m_sum   = 0;
        const int d = c_u_max[v];

        for (int u = threadIdx.x - d; u <= d; u += blockDim.x)
        {
            //            int val_plus  = image(loc.y + v, loc.x + u);
            //            int val_minus = image(loc.y - v, loc.x + u);

            int val_plus  = tex2D<unsigned char>(tex, loc.x + u, loc.y + v);
            int val_minus = tex2D<unsigned char>(tex, loc.x + u, loc.y - v);

            v_sum += (val_plus - val_minus);
            m_sum += u * (val_plus + val_minus);
        }

        m_sum = Saiga::CUDA::warpReduceSum<int, 32, false>(m_sum);
        v_sum = Saiga::CUDA::warpReduceSum<int, 32, false>(v_sum);

        m_10 += m_sum;
        m_01 += v * v_sum;
    }

    if (threadIdx.x == 0)
    {
        float kp_dir = atan2((float)m_01, (float)m_10);
        kp_dir += (kp_dir < 0) * (2.0f * float(pi<float>()));
        kp_dir *= 180.0f / float(pi<float>());

        keypoints[ptidx].angle = kp_dir;
    }
}

__global__ void addBorder_kernel(Saiga::KeyPoint<float>* keypoints, int npoints, int minBorderX, int minBorderY,
                                 int octave, int size)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= npoints)
    {
        return;
    }
    keypoints[tid].point(0) += minBorderX;
    keypoints[tid].point(1) += minBorderY;
    keypoints[tid].octave = octave;
    keypoints[tid].size   = size;
}

void ORB::ComputeAngles(cudaTextureObject_t tex, Saiga::ImageView<unsigned char> image,
                        Saiga::ArrayView<Saiga::KeyPoint<float>> _keypoints, int minBorderX, int minBorderY, int octave,
                        int size, cudaStream_t stream)
{
    if (_keypoints.empty())
    {
        return;
    }

    {
        dim3 block(256);
        dim3 grid(Saiga::iDivUp<int>(_keypoints.size(), block.x));
        addBorder_kernel<<<grid, block, 0, stream>>>(_keypoints.data(), _keypoints.size(), minBorderX, minBorderY,
                                                     octave, size);
    }
    {
        dim3 block(32, 8);
        dim3 grid(Saiga::iDivUp<int>(_keypoints.size(), block.y));
        IC_Angle_kernel<<<grid, block, 0, stream>>>(tex, image, _keypoints);
    }
}


}  // namespace CUDA
}  // namespace Saiga

#endif
