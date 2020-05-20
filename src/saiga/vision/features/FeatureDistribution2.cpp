/**
 * This file is part of ORB-SLAM2.
 * This file is based on the file orb.cpp from the OpenCV library (see BSD license below).
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

#include "FeatureDistribution2.h"

#include "saiga/core/time/all.h"
#include "saiga/core/util/assert.h"


namespace Saiga
{
std::array<Distributor::QuadtreeNode, 4> Distributor::QuadtreeNode::splitAndSort(
    ArrayView<KeyPoint<float>> keypoints, std::array<std::vector<KeyPoint<float>>, 4>& local_kps)
{
    vec2 new_size = size * 0.5f;
    vec2 center   = corner + new_size;


    for (auto& kps : local_kps)
    {
        kps.clear();
    }

    for (auto i : Range<int>(from, to))
    {
        auto& kp = keypoints[i];

        int less_x = kp.point.x() > center.x();
        int less_y = kp.point.y() > center.y();


        int idx = (less_y << 1) + less_x;

        SAIGA_ASSERT(local_kps[idx].size() <= local_kps[idx].capacity());
        local_kps[idx].push_back(kp);
    }


    std::array<QuadtreeNode, 4> result;


    int count          = 0;
    int previous_count = 0;
    for (int k : Range<int>(0, 4))
    {
        for (auto& kp : local_kps[k])
        {
            keypoints[from + count] = kp;
            count++;
        }

        result[k].from = from + previous_count;
        result[k].to   = from + count;
        previous_count = count;

        result[k].size = new_size;
    }

    SAIGA_ASSERT(count == to - from);

    result[0].corner = vec2(corner(0), corner(1));
    result[1].corner = vec2(center(0), corner(1));
    result[2].corner = vec2(corner(0), center(1));
    result[3].corner = vec2(center(0), center(1));

    return result;
}

std::vector<KeyPoint<float>> Distributor::DistributeOctTree(ArrayView<KeyPoint<float>> keypoints,
                                                            const vec2& min_position, const vec2& max_position,
                                                            int target_n)
{
    inner_nodes.clear();
    new_inner_nodes.clear();
    leaf_nodes.clear();
    new_inner_nodes.reserve(target_n * 4);
    inner_nodes.reserve(target_n * 4);
    leaf_nodes.reserve(target_n * 4);

    if (keypoints.size() <= target_n)
    {
        return {keypoints.begin(), keypoints.end()};
    }


    for (auto& v : local_kps)
    {
        v.clear();
        v.reserve(keypoints.size());
    }

    //    SAIGA_BLOCK_TIMER();
    vec2 center       = (min_position + max_position) * 0.5f;
    vec2 size         = max_position - min_position;
    float max_size    = std::max(size(0), size(1));
    vec2 start_corner = center - vec2(max_size, max_size) * 0.5f;



    new_inner_nodes.emplace_back(QuadtreeNode(start_corner, vec2(max_size, max_size), 0, keypoints.size()));
    //    inner_nodes.emplace_back(QuadtreeNode(vec2(0, 0), vec2(size), 0, keypoints.size()));
    //    inner_nodes.emplace_back(QuadtreeNode(vec2(0, 0), vec2(size(0), size(1) * 2), 0, keypoints.size()));
    int last_node_count = 0;

    int last_processed_inner = -1;
    bool bFinish             = false;

    while (!bFinish && !new_inner_nodes.empty())
    {
        inner_nodes.swap(new_inner_nodes);
        new_inner_nodes.clear();

        int node_count = leaf_nodes.size() + inner_nodes.size();
        if (node_count == last_node_count)
        {
            break;
        }
        last_node_count = node_count;

        // sort inner nodes by size
        std::sort(inner_nodes.begin(), inner_nodes.end(),
                  [](const auto& n1, const auto& n2) { return n1.Size() > n2.Size(); });

        // split until we have enough
        for (int i = 0; i < inner_nodes.size(); ++i)
        {
            auto& node = inner_nodes[i];

            SAIGA_ASSERT(node.Size() > 1);
            auto result = node.splitAndSort(keypoints, local_kps);

            for (auto& r : result)
            {
                auto size = r.Size();
                if (size == 0)
                {
                    continue;
                }
                else if (size == 1)
                {
                    leaf_nodes.push_back(r);
                }
                else
                {
                    new_inner_nodes.push_back(r);
                }
            }

            int remaining_inner_nodes = inner_nodes.size() - i - 1;
            if (leaf_nodes.size() + new_inner_nodes.size() + remaining_inner_nodes >= target_n)
            {
                last_processed_inner = i;
                bFinish              = true;
                break;
            }
        }
    }


    std::vector<KeyPoint<float>> result;
    result.reserve(leaf_nodes.size() + inner_nodes.size());

    auto add_best_to_result = [&](const auto& node) {
        auto best = std::max_element(keypoints.begin() + node.from, keypoints.begin() + node.to,
                                     [](const auto& kp1, const auto& kp2) { return kp1.response < kp2.response; });
        result.push_back(*best);
    };

    for (auto& node : leaf_nodes)
    {
        add_best_to_result(node);
    }

    for (auto& node : new_inner_nodes)
    {
        add_best_to_result(node);
    }


    for (int i = last_processed_inner; i < inner_nodes.size(); ++i)
    {
        add_best_to_result(inner_nodes[i]);
    }

    return result;
}

int TemporalKeypointFilter::Filter(int image_height, int image_width, ArrayView<KeyPoint<float>> keypoints)
{
    if (noise_.rows() != image_height || noise_.cols() != image_width)
    {
        noise_.resize(image_height, image_width);
        noise_.setZero();
    }

    int N                = keypoints.size();
    int noisy_points     = 0;
    int max_noisy_points = N;
    for (int i = 0; i < N; ++i)
    {
        auto& kp = keypoints[i];
        // low-response points might be created due to sensor noise
        if (kp.response < response_threshold_)
        {
            auto& noise = noise_(int(kp.point.y()), int(kp.point.x()));
            noise += 1;
            if (noisy_points < max_noisy_points && noise > noise_threshold_)
            {
                keypoints[i] = keypoints.back();
                keypoints.pop_back();

                N--;
                i--;
                noisy_points++;
            }
        }
    }
    return N;
}

int TemporalKeypointFilter::FilterRescale(int image_height, int image_width, const vec2& border, float scale,
                                          ArrayView<KeyPoint<float>> keypoints)
{
    if (noise_.rows() != image_height || noise_.cols() != image_width)
    {
        noise_.resize(image_height, image_width);
        noise_.setZero();
    }

    int N                = keypoints.size();
    int noisy_points     = 0;
    int max_noisy_points = 100;
    for (int i = 0; i < N; ++i)
    {
        auto& kp = keypoints[i];
        // low-response points might be created due to sensor noise
        if (kp.response < response_threshold_)
        {
            vec2 point = (kp.point + border) * scale;
            int px     = int(point.x());
            int py     = int(point.y());
            SAIGA_ASSERT(px >= 0 && px < noise_.cols());
            SAIGA_ASSERT(py >= 0 && py < noise_.rows());
            auto& noise = noise_(py, px);
            noise += 1;
            if (noisy_points < max_noisy_points && noise > noise_threshold_)
            {
                keypoints[i] = keypoints.back();
                keypoints.pop_back();

                N--;
                i--;
                noisy_points++;
            }
        }
    }
    return N;
}

void TemporalKeypointFilter::Step()
{
    noise_ *= alpha_;
}


}  // namespace Saiga
