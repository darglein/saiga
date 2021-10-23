/**
 * This file is part of ORB-SLAM2.
 * This file is based on the file orb.cpp from the OpenCV library (see BSD
 * license below).
 *
 * Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University
 * of Zaragoza) For more information see <https://github.com/raulmur/ORB_SLAM2>
 *
 * ORB-SLAM2 is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ORB-SLAM2 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
 */
/**
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2009, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the Willow Garage nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#include "saiga/core/time/all.h"
#include "saiga/core/util/Thread/omp.h"
#include "saiga/core/util/assert.h"
#include "saiga/cuda/cudaHelper.h"

#ifdef SAIGA_VISION
#include "ORBExtractorGPU.h"

#include <thread>
#include <vector>

#if defined(SAIGA_USE_CUDA_TOOLKIT) && !defined(_WIN32)
const int PATCH_SIZE = 31;

namespace Saiga
{
ORBExtractorGPU::Level::Level() {}
ORBExtractorGPU::Level::~Level() {}
void ORBExtractorGPU::Level::Reserve(int initial_N, int final_N)
{
    descriptors.resize(final_N);
    h_descriptors.resize(final_N);
    keypoints.resize(initial_N);
    h_keypoints.resize(initial_N);
}

void ORBExtractorGPU::Level::filter()
{
    SAIGA_ASSERT(image.rows == image_gauss.rows);
    SAIGA_ASSERT(image.cols == image_gauss.cols);
    Saiga::NPPI::GaussFilter(image.getConstImageView(), image_gauss.getImageView(), context);
}

ORBExtractorGPU::ORBExtractorGPU(int _nfeatures, float _scaleFactor, int _nlevels, int _iniThFAST, int _minThFAST)
    : nlevels(_nlevels), iniThFAST(_iniThFAST), minThFAST(_minThFAST)

{
    pyramid = Saiga::ScalePyramid(_nlevels, _scaleFactor, _nfeatures);

    download_stream.setName("download");
    orb_stream.setName("orb");
    descriptor_stream.setName("descritpor");
}

ORBExtractorGPU::~ORBExtractorGPU() {}

int ORBExtractorGPU::Detect(Saiga::ImageView<unsigned char> image, std::vector<Saiga::KeyPoint<float>>& keypoints,
                            std::vector<Saiga::DescriptorORB>& descriptors)
{
    //    SAIGA_BLOCK_TIMER();

    SAIGA_ASSERT(!image.empty());
    SAIGA_ASSERT(image.pitchBytes % 4 == 0);

    image.cols = Saiga::iAlignDown(image.cols, 4);

    ComputePyramid(image);

    int total_kps = 0;
    for (auto& level : levels)
    {
        total_kps += level->N;
    }

    keypoints.resize(total_kps);
    descriptors.resize(total_kps);

    ComputeKeypoints(keypoints, descriptors);

    return total_kps;
}

void ORBExtractorGPU::DownloadAndDistribute(int level)
{
    auto& level_data = levels[level];

    {
        SAIGA_ASSERT(level_data->N <= level_data->h_keypoints.size());
        auto h_keypoints = Saiga::ArrayView<Saiga::KeyPoint<float>>(level_data->h_keypoints).head(level_data->N);

        auto level_keypoints = level_data->dis.Distribute(
            h_keypoints, Saiga::vec2(level_data->fast_min_x, level_data->fast_min_y),
            Saiga::vec2(level_data->fast_max_x, level_data->fast_max_y), pyramid.Features(level));

        SAIGA_ASSERT(level_keypoints.size() <= h_keypoints.size());

        level_data->N = level_keypoints.size();
        h_keypoints  = h_keypoints.head(level_data->N);

        for (int i = 0; i < level_data->N; ++i)
        {
            h_keypoints[i] = level_keypoints[i];
        }
    }

    {
        const int N = level_data->N;

        SAIGA_ASSERT(level_data->h_keypoints.size() >= N);
        SAIGA_ASSERT(level_data->keypoints.size() >= N);
        SAIGA_ASSERT(level_data->h_descriptors.size() >= N);
        SAIGA_ASSERT(level_data->descriptors.size() >= N);

        auto h_keypoints   = Saiga::ArrayView<Saiga::KeyPoint<float>>(level_data->h_keypoints).head(N);
        auto d_keypoints   = Saiga::ArrayView<Saiga::KeyPoint<float>>(level_data->keypoints).head(N);
        auto h_descriptors = Saiga::ArrayView<Saiga::DescriptorORB>(level_data->h_descriptors).head(N);
        auto d_descriptors = Saiga::ArrayView<Saiga::DescriptorORB>(level_data->descriptors).head(N);

        auto& stream = level_data->stream;

        CHECK_CUDA_ERROR(cudaMemcpyAsync(d_keypoints.data(), h_keypoints.data(), sizeof(Saiga::KeyPoint<float>) * N,
                                         cudaMemcpyHostToDevice, stream));

        {
            orb.ComputeAngles(level_data->image_obj, level_data->image.getImageView(), d_keypoints, level_data->fast_min_x,
                              level_data->fast_min_y, level, pyramid.Scale(level) * PATCH_SIZE, stream);
        }

        {
#    ifdef SAIGA_NPPI_HAS_STREAM_CONTEXT
#    else
            level_data->gauss_ready.wait(stream);
#    endif

            orb.ComputeDescriptors(level_data->image_gauss_obj, level_data->image_gauss.getImageView(), d_keypoints,
                                   d_descriptors, stream);
        }

        CHECK_CUDA_ERROR(cudaMemcpyAsync(h_keypoints.data(), d_keypoints.data(), sizeof(Saiga::KeyPoint<float>) * N,
                                         cudaMemcpyDeviceToHost, stream));
        CHECK_CUDA_ERROR(cudaMemcpyAsync(h_descriptors.data(), d_descriptors.data(), sizeof(Saiga::DescriptorORB) * N,
                                         cudaMemcpyDeviceToHost, stream));
    }
}

void ORBExtractorGPU::ComputeKeypoints(std::vector<Saiga::KeyPoint<float>>& keypoints,
                                       std::vector<Saiga::DescriptorORB>& descriptors)
{
    int current_kp = 0;
    for (int level = 0; level < nlevels; ++level)
    {
        auto& level_data   = levels[level];
        int N              = level_data->N;
        auto h_keypoints   = Saiga::ArrayView<Saiga::KeyPoint<float>>(level_data->h_keypoints).head(N);
        auto h_descriptors = Saiga::ArrayView<Saiga::DescriptorORB>(level_data->h_descriptors).head(N);

        level_data->stream.synchronize();

        float scale = pyramid.Scale(level);
        for (int i = 0; i < N; ++i)
        {
            auto& kp = h_keypoints[i];
            kp.point *= scale;
            keypoints[current_kp + i]   = kp;
            descriptors[current_kp + i] = h_descriptors[i];
        }
        current_kp += N;
    }
}

void ORBExtractorGPU::AllocatePyramid(int rows, int cols)
{
    SAIGA_ASSERT(cols % 4 == 0);

    if (!levels.empty())
    {
        return;
    }

    levels.resize(nlevels);

    // first frame, allocate the Pyramids
    for (int level = 0; level < nlevels; ++level)
    {
        levels[level]    = std::make_shared<Level>();
        auto& level_data = levels[level];

        float scale    = pyramid.InverseScale(level);
        int level_rows = Saiga::iRound(rows * scale);
        int level_cols = Saiga::iRound(cols * scale);

        level_data->image.create(level_rows, level_cols);
        level_data->image_gauss.create(level_rows, level_cols);

        level_data->fast = std::make_unique<Saiga::CUDA::Fast>(iniThFAST, minThFAST);
        level_data->Reserve(level_data->fast->MaxKeypoints(), pyramid.Features(level) * 1.1);

        auto fast_edge_threshold = 16;
        level_data->fast_min_x    = fast_edge_threshold;
        level_data->fast_min_y    = fast_edge_threshold;
        level_data->fast_max_x    = level_data->image.cols - fast_edge_threshold;
        level_data->fast_max_y    = level_data->image.rows - fast_edge_threshold;

        level_data->fast_image_view = level_data->image.getImageView().subImageView(
            level_data->fast_min_y, level_data->fast_min_x, level_data->fast_max_y - level_data->fast_min_y,
            level_data->fast_max_x - level_data->fast_min_x);

        level_data->stream.setName("Level " + std::to_string(level));

        level_data->context = Saiga::NPPI::CreateStreamContextWithStream(level_data->stream);

        level_data->image_obj       = level_data->image.GetTextureObject();
        level_data->image_gauss_obj = level_data->image_gauss.GetTextureObject();
    }
    nppSetStream(orb_stream);
}

void ORBExtractorGPU::ComputePyramid(Saiga::ImageView<unsigned char> image)
{
    //    SAIGA_BLOCK_TIMER();
    AllocatePyramid(image.rows, image.cols);

    SAIGA_ASSERT(!levels.empty());

    auto& first_level = levels.front();

    SAIGA_ASSERT(first_level->image.cols == image.cols);
    SAIGA_ASSERT(first_level->image.rows == image.rows);

    for (int level = 0; level < nlevels; ++level)
    {
        auto& curr_data = levels[level];
        curr_data->image_ready.reset();
    }

    for (int level = 0; level < nlevels; ++level)
    {
        auto& curr_data = levels[level];

#    ifdef SAIGA_NPPI_HAS_STREAM_CONTEXT
        auto& stream = curr_data->stream;
#    else
        auto& stream = orb_stream;
#    endif

        if (level == 0)
        {
            first_level->image.upload(image, stream);
        }
        else
        {
            auto& prev_data = levels[level - 1];

            stream.waitForEvent(prev_data->image_ready);

            Saiga::NPPI::ResizeLinear(prev_data->image.getConstImageView(), curr_data->image.getImageView(),
                                      curr_data->context);
        }
        curr_data->image_ready.record(stream);
        curr_data->fast->Detect(curr_data->fast_image_view, stream);
    }

#    ifndef _OPENMP
#        error asdf
#    endif

#    pragma omp parallel for num_threads(2) schedule(static, 1)
    for (int level = 0; level < nlevels; ++level)
    {
        auto& curr_data = levels[level];

        curr_data->download();
#    ifdef SAIGA_NPPI_HAS_STREAM_CONTEXT
        curr_data->filter();
        DownloadAndDistribute(level);
#    else
        if (level == 0)
        {
            for (int l2 = 0; l2 < nlevels; l2 += 1)
            {
                auto& l = levels[l2];
                l.filter();
                l.gauss_ready.record(orb_stream);

                l.stream.waitForEvent(l.gauss_ready);
            }
        }
        DownloadAndDistribute(level);
#    endif
    }
}
}  // namespace Saiga

#endif

#endif
