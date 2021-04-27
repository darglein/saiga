/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "FeatureDistribution.h"

#include "saiga/core/time/all.h"
#include "saiga/core/util/assert.h"


namespace Saiga
{
std::array<QuadtreeFeatureDistributor::QuadtreeNode, 4> QuadtreeFeatureDistributor::QuadtreeNode::splitAndSort(
    ArrayView<KeyPoint<float>> keypoints, std::array<std::vector<KeyPoint<float>>, 4>& local_kps)
{
    SAIGA_ASSERT(from <= to);
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
        SAIGA_ASSERT(idx >= 0 && idx < 4);
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

std::vector<KeyPoint<float>> QuadtreeFeatureDistributor::Distribute(ArrayView<KeyPoint<float>> keypoints,
                                                                    const vec2& min_position, const vec2& max_position,
                                                                    int target_n)
{
    inner_nodes.clear();
    new_inner_nodes.clear();
    leaf_nodes.clear();
    new_inner_nodes.reserve(target_n * 4);
    inner_nodes.reserve(target_n * 4);
    leaf_nodes.reserve(target_n * 4);

    if ((int)keypoints.size() <= target_n)
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
                  [](const auto& n1, const auto& n2) { return n1.NumKeypoints() > n2.NumKeypoints(); });

        // split until we have enough
        for (int i = 0; i < (int)inner_nodes.size(); ++i)
        {
            auto& node = inner_nodes[i];

            SAIGA_ASSERT(node.NumKeypoints() > 1);
            auto result = node.splitAndSort(keypoints, local_kps);

            for (auto& r : result)
            {
                auto size = r.NumKeypoints();
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
            if ((int)leaf_nodes.size() + (int)new_inner_nodes.size() + remaining_inner_nodes >= target_n)
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


    for (size_t i = last_processed_inner; i < inner_nodes.size(); ++i)
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
