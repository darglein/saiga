/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/DataStructures/ArrayView.h"
#include "saiga/vision/features/Features.h"

#include <vector>

namespace Saiga
{
// Selects N keypoints from an input array of M keypoints with M >> N.
// The keypoints are selected so that they are evenly distributed over the image and keypoints with a high response are
// preferred. This class implements a quadtree-based algorithm similar to the algorithm found in ORB-SLAM2. Differences
// are in the way how the remaining keypoints are selected and the impolementation itself. This implemenation is around
// 2x more efficient than the reference impl. of ORB-SLAM2.
//
// This class is not thread safe, because it uses local variables during execution!!!
// Use one object of this class for each thread!
class SAIGA_VISION_API QuadtreeFeatureDistributor
{
   public:
    QuadtreeFeatureDistributor() = default;
    std::vector<Saiga::KeyPoint<float>> Distribute(ArrayView<KeyPoint<float>> keypoints, const vec2& min_position,
                                                   const vec2& max_position, int target_n);

   private:
    class QuadtreeNode
    {
       public:
        QuadtreeNode() {}
        QuadtreeNode(const vec2& corner, const vec2& size, int from = -1, int to = -1)
            : corner(corner), size(size), from(from), to(to)
        {
        }
        std::array<QuadtreeNode, 4> splitAndSort(ArrayView<KeyPoint<float>> keypoints,
                                                 std::array<std::vector<KeyPoint<float>>, 4>& local_kps);
        int NumKeypoints() const { return to - from; }

        // The other opposite corner is at (corner.x + size, corner.y + size)
        vec2 corner;
        vec2 size;
        int from, to;
    };

    std::vector<QuadtreeNode> leaf_nodes;
    std::vector<QuadtreeNode> inner_nodes;
    std::vector<QuadtreeNode> new_inner_nodes;
    std::array<std::vector<KeyPoint<float>>, 4> local_kps;
};

// Temporal keypoint filter to remove keypoints at the exact same image coordinates over multiple frames. Such keypoints
// are often a result from image noise.
class TemporalKeypointFilter
{
   public:
    TemporalKeypointFilter(float response_threshold = 20, float noise_threshold = 3, float alpha = 0.9)
        : response_threshold_(response_threshold), noise_threshold_(noise_threshold), alpha_(alpha)
    {
    }

    int Filter(int image_height, int image_width, ArrayView<KeyPoint<float>> keypoints);
    int FilterRescale(int image_height, int image_width, const vec2& border, float scale,
                      ArrayView<KeyPoint<float>> keypoints);
    void Step();

   private:
    float response_threshold_;
    float noise_threshold_;
    float alpha_;
    Eigen::Array<float, Eigen::Dynamic, Eigen::Dynamic> noise_;
};

}  // namespace Saiga
