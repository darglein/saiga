/**
 * This file is part of ORB-SLAM2.
 *
 * Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
 * For more information see <https://github.com/raulmur/ORB_SLAM2>
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
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 *
 * This implemntation is completely reworked, but is based on the ORB-SLAM2 implementation.
 * See License above.
 */
#pragma once

#include "saiga/config.h"
#include "saiga/core/image/imageView.h"
#include "saiga/core/image/templatedImage.h"
#include "saiga/vision/features/FeatureDistribution.h"
#include "saiga/vision/features/Features.h"
#include "saiga/vision/features/OrbDescriptors.h"
#include "saiga/vision/util/ScalePyramid.h"

#include <vector>

#ifdef SAIGA_USE_OPENCV

namespace Saiga
{
class SAIGA_VISION_API ORBExtractor
{
   public:
    using KeypointType = Saiga::KeyPoint<float>;

    ORBExtractor(int nfeatures, float scaleFactor, int num_levels, int th_fast, int th_fast_min, int num_threads);
    ~ORBExtractor() {}

    void Detect(Saiga::ImageView<unsigned char> inputImage, std::vector<KeypointType>& keypoints,
                std::vector<Saiga::DescriptorORB>& outputDescriptors);

    // Can be called after 'Detect' to return the scaled image on the given level.
    // The imageview is invalidated after calling detect again.
    ImageView<unsigned char> GetImage(int level){
        return levels[level].image;
    }

    ScalePyramid getPyramid(){
        return pyramid;
    }

   protected:
    void AllocatePyramid(int rows, int cols);
    void ComputePyramid(Saiga::ImageView<unsigned char> image);
    void DetectKeypoints();

    int num_levels;
    int th_fast;
    int th_fast_min;
    int num_threads;

    Saiga::ORB orb;
    Saiga::ScalePyramid pyramid;


    struct Level
    {
        int N;
        int offset;
        Saiga::TemplatedImage<unsigned char> image_with_border;
        Saiga::TemplatedImage<unsigned char> image_gauss;
        Saiga::ImageView<unsigned char> image;
        std::vector<KeypointType> keypoints_tmp;
        Saiga::QuadtreeFeatureDistributor distributor;
    };
    std::vector<Level> levels;
};

}  // namespace Saiga

#endif
