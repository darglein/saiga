/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "RectangularDecomposition.h"

#include "saiga/core/math/Morton.h"
#include "saiga/core/math/random.h"
#include "saiga/core/time/all.h"

#include "RectilinearOptimization.h"

#include <iomanip>

constexpr bool verbose = false;


namespace Saiga
{
namespace RectangularDecomposition
{
std::vector<ivec3> RemoveDuplicates(ArrayView<const ivec3> points)
{
    std::vector<ivec3> result(points.begin(), points.end());
    std::sort(result.begin(), result.end(),
              [](auto a, auto b) { return std::tie(a(0), a(1), a(2)) < std::tie(b(0), b(1), b(2)); });
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}



RectangleList DecomposeTrivial(ArrayView<const ivec3> points)
{
    if (points.empty()) return {};
    RectangleList result;
    for (auto p : points)
    {
        result.push_back(Rect(p));
    }
    return result;
}

RectangleList DecomposeRowMerge(ArrayView<const ivec3> points)
{
    if (points.empty()) return {};
    int dim = 1;

    int s1 = (dim + 1) % 3;
    int s2 = (dim + 2) % 3;
    int s3 = dim;


    RectangleList result;
    if (points.empty()) return result;


    std::vector<ivec3> copy(points.begin(), points.end());
    std::sort(copy.begin(), copy.end(),
              [=](auto a, auto b) { return std::tie(a(s1), a(s2), a(s3)) < std::tie(b(s1), b(s2), b(s3)); });


    Rect current = Rect(copy.front());


    for (int i = 1; i < (int)copy.size(); ++i)
    {
        auto index = copy[i];

        ivec3 offset = ivec3::Zero();
        offset(s3)   = current.Size()(s3);

        if (index == current.begin + offset)
        {
            current.end(s3) += 1;
        }
        else
        {
            result.push_back(current);
            current = Rect(index);
        }
    }
    result.push_back(current);

    return result;
}


RectangleList DecomposeOctTree(ArrayView<const ivec3> points, float merge_factor, bool merge_layer)
{
    SAIGA_OPTIONAL_BLOCK_TIMER(verbose);
    RectangleList result;
    if (points.empty()) return result;

    ivec3 corner = points.front();
    for (auto& p : points)
    {
        corner = corner.array().min(p.array());
    }


    std::vector<std::pair<uint64_t, Rect>> copy, merged_list;

    for (auto p : points)
    {
        ivec3 shifted_p = p - corner;
        copy.emplace_back(Morton3D(shifted_p), shifted_p);
    }

    std::sort(copy.begin(), copy.end(), [](auto a, auto b) { return a.first < b.first; });



    //     for (int bit = 3; bit < 64; bit += 3)
    for (int bit = 1; bit < 64; bit += 1)
    {
        //        std::cout << "it " << bit << " " << copy.size() << " " << merged_list.size() << std::endl;
        uint64_t mask = (~0UL) << bit;
        //    std::cout << std::hex << mask << std::endl;


        int range_begin = 0;
        int range_end   = 0;

        for (int i = 0; i < (int)copy.size(); ++i)
        {
            auto& prev = copy[range_begin];
            auto& curr = copy[i];

            //            std::cout << i << ", " << (curr.first & mask) << " | " << copy[i].second << std::endl;
            // check if prev an current have same morton code except last bits



            //            std::cout << (prev.first & mask) << " " << (curr.first & mask) << std::endl;

            if ((prev.first & mask) == (curr.first & mask) && i != (int)copy.size() - 1)
            {
                //                std::cout << "merge " << bit << " " << prev.second << " with " << curr.second <<
                //                std::endl;
                //                merged_list.push_back({prev.first & mask, Rect(prev.second, curr.second)});
                //                prev.second.end = prev.second.begin;
                //                curr.second.end = curr.second.begin;
                //                i++;
                range_end++;
            }
            else
            {
                if (i == (int)copy.size() - 1)
                {
                    range_end++;
                }

                // std::cout << "merge " << range_end - range_begin << std::endl;

                // Let's merge all elements in the range
                Rect merged_range = copy[range_begin].second;
                for (auto j = range_begin + 1; j < range_end; ++j)
                {
                    merged_range = Rect(merged_range, copy[j].second);
                }



                int volume_sum = 0;
                for (int j = range_begin; j < range_end; ++j)
                {
                    volume_sum += copy[j].second.Volume();
                }



                float factor = volume_sum / float(merged_range.Volume());
                if (factor >= merge_factor)
                {
                    merged_list.push_back({prev.first & mask, merged_range});
                    for (int j = range_begin; j < range_end; ++j)
                    {
                        copy[j].second.setZero();
                    }
                }
#if 0
                else
                {
                    for (int j = range_begin; j < range_end; ++j)
                    {
                        merged_list.push_back(copy[j]);
                        copy[j].second.setZero();
                    }
                }
#endif


                range_begin = i;
                range_end   = range_begin + 1;
            }
        }

        for (auto& c : copy)
        {
            if (c.second.Volume() > 0)
            {
                result.push_back(c.second);
            }
        }

        copy = merged_list;
        merged_list.clear();
        //        std::cout << std::endl;
    }

    for (auto& ml : copy)
    {
        result.push_back(ml.second);
    }


    for (auto& ml : merged_list)
    {
        result.push_back(ml.second);
    }


    for (auto& r : result)
    {
        r.begin += corner;
        r.end += corner;
    }


    //    return result;
    //    MergeNeighborsSave(result);
    return result;
}


}  // namespace RectangularDecomposition
}  // namespace Saiga
