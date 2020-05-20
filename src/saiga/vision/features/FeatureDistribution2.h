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

#pragma once


#include "saiga/core/image/all.h"
#include "saiga/vision/features/Features.h"

#include <vector>



namespace Saiga
{
class SAIGA_VISION_API Distributor
{
   public:
    Distributor() = default;
    std::vector<Saiga::KeyPoint<float>> DistributeOctTree(ArrayView<KeyPoint<float>> keypoints,
                                                          const vec2& min_position, const vec2& max_position,
                                                          int target_n);

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
        int Size() const { return to - from; }

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
