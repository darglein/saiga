/**
 * This file is part of ORB-SLAM2.
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

#pragma once

#include "saiga/cuda/imageProcessing/Fast.h"
#include "saiga/cuda/imageProcessing/NppiHelper.h"
#include "saiga/cuda/imageProcessing/OrbDescriptors.h"
#include "saiga/cuda/imageProcessing/image.h"
#include "saiga/cuda/pinned_vector.h"
#include "saiga/vision/features/FeatureDistribution.h"
#include "saiga/vision/util/ScalePyramid.h"

#if defined(SAIGA_USE_CUDA_TOOLKIT) && !defined(_WIN32)

namespace Saiga
{
class SAIGA_CUDA_API ORBExtractorGPU
{
   public:
    ORBExtractorGPU(int nfeatures, float scaleFactor, int nlevels, int iniThFAST, int minThFAST);

    ~ORBExtractorGPU();

    // Compute the ORB features and descriptors on an image.
    // ORB are dispersed on the image using an octree.
    // Mask is ignored in the current implementation.
    int Detect(Saiga::ImageView<unsigned char> image, std::vector<Saiga::KeyPoint<float>>& keypoints,
               std::vector<Saiga::DescriptorORB>& descriptors);

   private:
    // Pyramid allocation is not done in the constructor. It is deferred until the
    // first image arrives.
    void AllocatePyramid(int rows, int cols);
    void ComputePyramid(Saiga::ImageView<unsigned char> image);
    void ComputeKeypoints(std::vector<Saiga::KeyPoint<float>>& keypoints,
                          std::vector<Saiga::DescriptorORB>& descriptors);

    void DownloadAndDistribute(int level);
    struct Level
    {
        int N;
        int fast_min_x, fast_min_y;
        int fast_max_x, fast_max_y;
        Saiga::ImageView<unsigned char> fast_image_view;

        Saiga::CUDA::CudaImage<unsigned char> image;
        Saiga::CUDA::CudaImage<unsigned char> image_gauss;

        cudaTextureObject_t image_obj, image_gauss_obj;

        std::unique_ptr<Saiga::CUDA::Fast> fast;

        void Reserve(int initial_N, int final_N);

        Saiga::pinned_vector<Saiga::KeyPoint<float>> h_keypoints;
        thrust::device_vector<Saiga::KeyPoint<float>> keypoints;
        thrust::device_vector<Saiga::DescriptorORB> descriptors;
        Saiga::pinned_vector<Saiga::DescriptorORB> h_descriptors;

        Saiga::CUDA::CudaStream stream;
        Saiga::CUDA::CudaEvent image_ready;
        Saiga::CUDA::CudaEvent gauss_ready;
        SaigaNppStreamContext context;

        Saiga::QuadtreeFeatureDistributor dis;

        void download() { N = fast->Download(h_keypoints, stream); }

        void filter();

        Level();
        ~Level();
        // Level (Level& other) = delete;
        // Level& operator=(Level& other) = delete;
    };

    Saiga::CUDA::ORB orb;
    std::vector<std::shared_ptr<Level>> levels;

    std::vector<Saiga::ivec2> pattern;

    Saiga::CUDA::CudaStream download_stream;
    Saiga::CUDA::CudaStream orb_stream;
    Saiga::CUDA::CudaStream descriptor_stream;

    int nlevels;
    int iniThFAST;
    int minThFAST;

    Saiga::ScalePyramid pyramid;
};

}  // namespace Saiga

#endif
