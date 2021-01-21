/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "RectangularDecomposition.h"

#include "saiga/core/math/random.h"
#include "saiga/core/time/all.h"

#include "RectilinearOptimization.h"

#include <iomanip>



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


bool Decomposition::ContainsAll(ArrayView<const ivec3> points) const
{
    for (auto& p : points)
    {
        bool found = false;
        for (auto& rect : rectangles)
        {
            if (rect.Contains(p))
            {
                found = true;
                break;
            }
        }

        if (!found)
        {
            return false;
        }
    }
    return true;
}

Decomposition Decomposition::RemoveFullyContained() const
{
    PointHashMap<3> map;
    for (auto& r : rectangles)
    {
        map.Add(r, 1);
    }


    Decomposition result;

    for (auto& r : rectangles)
    {
        map.Add(r, -1);
        if (map.AllGreater(r, 0))
        {
            //            std::cout << "remove " << r << std::endl;
        }
        else
        {
            result.rectangles.push_back(r);
        }
    }

    return result;
}

std::ostream& operator<<(std::ostream& strm, const Decomposition& decomp)
{
    strm << "[Decomp] N = " << std::setw(6) << decomp.rectangles.size() << "  V0 = " << std::setw(6) << decomp.Volume()
         << "  V1 = " << std::setw(6) << decomp.ExpandedVolume(1) << "  V2 = " << std::setw(6)
         << decomp.ExpandedVolume(2);
    return strm;
}

Decomposition TrivialRectangularDecomposition::Compute(ArrayView<const ivec3> points)
{
    if (points.empty()) return {};
    Decomposition result;
    for (auto p : points)
    {
        result.rectangles.push_back(Rect(p));
    }
    return result;
}

Decomposition RowMergeDecomposition::Compute(ArrayView<const ivec3> points)
{
    if (points.empty()) return {};
    int dim = 1;

    int s1 = (dim + 1) % 3;
    int s2 = (dim + 2) % 3;
    int s3 = dim;


    Decomposition result;
    if (points.empty()) return result;


    std::vector<ivec3> copy(points.begin(), points.end());
    std::sort(copy.begin(), copy.end(),
              [=](auto a, auto b) { return std::tie(a(s1), a(s2), a(s3)) < std::tie(b(s1), b(s2), b(s3)); });


    Rect current = Rect(copy.front());


    for (int i = 1; i < copy.size(); ++i)
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
            result.rectangles.push_back(current);
            current = Rect(index);
        }
    }
    result.rectangles.push_back(current);

    return result;
}

uint64_t BitInterleave64(uint64_t x)
{
    x &= 0x1fffff;
    x = (x | x << 32) & 0x1f00000000ffff;
    x = (x | x << 16) & 0x1f0000ff0000ff;
    x = (x | x << 8) & 0x100f00f00f00f00f;
    x = (x | x << 4) & 0x10c30c30c30c30c3;
    x = (x | x << 2) & 0x1249249249249249;
    return x;
}

uint64_t Morton2D(const ivec3& v)
{
    uint64_t x = BitInterleave64(v.x());
    uint64_t y = BitInterleave64(v.y());
    uint64_t z = BitInterleave64(v.z());
    return x | (y << 1) | (z << 2);
}

Decomposition OctTreeDecomposition::Compute(ArrayView<const ivec3> points)
{
    SAIGA_BLOCK_TIMER();
    Decomposition result;
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
        copy.emplace_back(Morton2D(shifted_p), shifted_p);
    }

    std::sort(copy.begin(), copy.end(), [](auto a, auto b) { return a.first < b.first; });



    for (int bit = 3; bit < 64; bit += 3)
    {
        //        std::cout << "it " << bit << " " << copy.size() << " " << merged_list.size() << std::endl;
        uint64_t mask = (~0UL) << bit;
        //    std::cout << std::hex << mask << std::endl;


        int range_begin = 0;
        int range_end   = 0;

        for (int i = 0; i < copy.size(); ++i)
        {
            auto& prev = copy[range_begin];
            auto& curr = copy[i];

            //            std::cout << i << ", " << (curr.first & mask) << " | " << copy[i].second << std::endl;
            // check if prev an current have same morton code except last bits



            //            std::cout << (prev.first & mask) << " " << (curr.first & mask) << std::endl;

            if ((prev.first & mask) == (curr.first & mask) && i != copy.size() - 1)
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
                if (i == copy.size() - 1)
                {
                    range_end++;
                }

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
                range_begin = i;
                range_end   = range_begin + 1;
            }
        }

        for (auto& c : copy)
        {
            if (c.second.Volume() > 0)
            {
                result.rectangles.push_back(c.second);
            }
        }

        copy = merged_list;
        merged_list.clear();
        //        std::cout << std::endl;
    }

    for (auto& ml : copy)
    {
        result.rectangles.push_back(ml.second);
    }


    for (auto& ml : merged_list)
    {
        result.rectangles.push_back(ml.second);
    }


    for (auto& r : result.rectangles)
    {
        r.begin += corner;
        r.end += corner;
    }


    //    return result;
    MergeNeighborsSave(result.rectangles);
    return result;
}

Decomposition SaveMergeDecomposition::Compute(ArrayView<const ivec3> points)
{
    SAIGA_BLOCK_TIMER();
    if (points.empty()) return {};
    OctTreeDecomposition octree;
    Decomposition result = octree.Compute(points);

    // First do a 'save' optimization without increasing v0
    auto tmp = cost;
    cost     = VolumeCost({0.1, 1});
    result   = Optimize(result);
    //    std::cout << result << std::endl;

    // Second do the actual optimization with the user defined cost
    cost   = tmp;
    result = Optimize(result);
    //    std::cout << result << std::endl;

    return result;
}

Decomposition SaveMergeDecomposition::Optimize(const Decomposition& decomp)
{
    SAIGA_BLOCK_TIMER("SaveMergeDecomposition::Optimize");
    Decomposition result = decomp;

    MergeNeighbors(result.rectangles, cost);
    return result;
}

Decomposition GrowAndShrinkDecomposition::Compute(ArrayView<const ivec3> points)
{
    SAIGA_BLOCK_TIMER();
    if (points.empty()) return {};
    {
        SaveMergeDecomposition triv;
        triv.cost             = cost;
        current_decomp        = triv.Compute(points);
        current_cost          = cost(current_decomp.rectangles);
        not_improved_in_a_row = 0;
    }

    {
        SAIGA_BLOCK_TIMER("GrowAndShrinkDecomposition iteration");
        for (int it = 0; it < its; ++it)
        {
            // find a grow that reduces the cost

            auto cpy = current_decomp;
            //                    RandomStepGrow(cpy.first);
            RandomStepMerge(cpy.rectangles);

            auto c = cost(cpy.rectangles);
            if (c < current_cost)
            {
                current_decomp        = cpy;
                current_cost          = c;
                not_improved_in_a_row = 0;
            }
            else
            {
                not_improved_in_a_row++;
            }

            if (not_improved_in_a_row == converge_its)
            {
                break;
            }
            std::cout << "It " << it << " " << current_decomp << " C = " << current_cost << std::endl;
        }
    }
    return current_decomp;
}

void GrowAndShrinkDecomposition::RandomStepMerge(RectangleList& rectangles)
{
    int ind = Random::uniformInt(0, rectangles.size() - 1);
    auto& r = rectangles[ind];

    std::vector<int> indices;
    for (int i = 0; i < rectangles.size(); ++i)
    {
        auto& r2 = rectangles[i];
        if (i != ind && r.Distance(r2) <= 2)
        {
            indices.push_back(i);
        }
    }

    if (indices.empty())
    {
        return;
    }

    auto& r2 = rectangles[indices[Random::uniformInt(0, indices.size() - 1)]];

    r  = Rect(r, r2);
    r2 = Rect();
    std::swap(rectangles.back(), rectangles[ind]);

    ShrinkIfPossible(rectangles);
}

void GrowAndShrinkDecomposition::RandomStepGrow(RectangleList& rectangles)
{
    int ind = Random::uniformInt(0, rectangles.size() - 1);

    int id = Random::uniformInt(0, 5);

    if (id < 3)
    {
        rectangles[ind].begin(id) -= 1;
    }
    else
    {
        rectangles[ind].end(id - 3) += 1;
    }

    // move to last spot
    std::swap(rectangles.back(), rectangles[ind]);

    ShrinkIfPossible(rectangles);
}



}  // namespace RectangularDecomposition
}  // namespace Saiga
