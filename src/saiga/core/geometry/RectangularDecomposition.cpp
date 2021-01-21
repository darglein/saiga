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

Decomposition Decomposition::ShrinkIfPossible() const
{
    PointHashMap<3> map;
    for (auto& r : rectangles)
    {
        map.Add(r, 1);
    }

    Decomposition result;
    result.rectangles.reserve(rectangles.size());

    for (auto& r : rectangles)
    {
        map.Add(r, -1);

        // init with empty
        Rect keep_rect = r;
        Rect removeable_rect(r.begin, r.begin);


        for (int axis = 0; axis < 3; ++axis)
        {
            for (int z = r.begin(axis); z <= r.end(axis); ++z)
            {
                Rect r1 = r;
                Rect r2 = r;

                r1.end(axis)   = z;
                r2.begin(axis) = z;

                if (removeable_rect.Volume() < r1.Volume() && map.AllGreater(r1, 0))
                {
                    removeable_rect = r1;
                    keep_rect       = r2;
                }

                if (removeable_rect.Volume() < r2.Volume() && map.AllGreater(r2, 0))
                {
                    removeable_rect = r2;
                    keep_rect       = r1;
                }
            }
        }
        if (keep_rect.Volume() > 0)
        {
            result.rectangles.push_back(keep_rect);
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

    bool changed = true;
    while (changed)
    {
        changed     = false;
        auto neighs = NeighborList(result.rectangles, 1);
        for (auto n : neighs)
        {
            auto& r1 = result.rectangles[n.first];
            if (r1.Volume() == 0) continue;
            auto& r2 = result.rectangles[n.second];
            if (r2.Volume() == 0) continue;

            SAIGA_ASSERT(!r1.Intersect(r2));

            // 1. Let's compute the merged rectangle and then check later if this merge is viable
            Rect merged = Rect(r1, r2);

            // 2. Compute all intersecting rects towards the new merged Rectangle
            auto inters = AllIntersectingRects(result.rectangles, merged);
            std::vector<std::tuple<Rect, Rect, Rect>> shrunk(inters.size());
            bool found = false;

            // 3. Check if all intersecting rects can be shrunk to the merged rect.
            for (int i = 0; i < inters.size(); ++i)
            {
                auto& r = result.rectangles[inters[i]];


                if (inters[i] == n.first)
                {
                    std::get<0>(shrunk[i]) = r1;
                    continue;
                }

                if (inters[i] == n.second)
                {
                    std::get<0>(shrunk[i]) = r2;
                    continue;
                }
                SAIGA_ASSERT(!r1.Intersect(r));
                SAIGA_ASSERT(!r2.Intersect(r));

                if (!merged.ShrinkOtherToThis(r, std::get<0>(shrunk[i]), std::get<1>(shrunk[i]),
                                              std::get<2>(shrunk[i])))
                {
                    found = true;
                    break;
                }
            }

            // We found one rect which cannot be shrunk -> abort
            if (found)
            {
                continue;
            }


            // 4. Compute merged volume and compute it to the transformed volume
            float old_cost = 0;
            for (int i = 0; i < inters.size(); ++i)
            {
                auto& r = result.rectangles[inters[i]];
                old_cost += cost(r);
            }

            // The new cost is the merged rectangle + all outer rects
            float new_cost = cost(merged);
            for (auto r : shrunk)
            {
                new_cost += cost(std::get<1>(r));
                new_cost += cost(std::get<2>(r));
            }



            // 5. If the merged colume is smaller or equal (equal is also good because we removed one rectangle),
            //    then apply the merge by setting r1 to the merged rect and all other intersections to the inner shrunk
            //    results.
            if (new_cost < old_cost)
            {
                for (int i = 0; i < inters.size(); ++i)
                {
                    auto& r = result.rectangles[inters[i]];

                    if (inters[i] == n.first)
                    {
                        r = merged;
                    }
                    else
                    {
                        r = std::get<1>(shrunk[i]);
                        result.rectangles.push_back(std::get<2>(shrunk[i]));
                    }
                }
                changed = true;
            }
        }
        if (changed)
        {
            RemoveEmpty(result.rectangles);
        }
    }

    return result;
}

Decomposition GrowAndShrinkDecomposition::Compute(ArrayView<const ivec3> points)
{
    SAIGA_BLOCK_TIMER();
    if (points.empty()) return {};
    {
        SaveMergeDecomposition triv;
        triv.cost = cost;
        auto d    = triv.Compute(points);

        decomps.push_back({d, cost(d.rectangles)});


        current_best          = Best().second;
        not_improved_in_a_row = 0;
    }

    {
        SAIGA_BLOCK_TIMER("GrowAndShrinkDecomposition iteration");
        for (int it = 0; it < its; ++it)
        {
            {
                //                SAIGA_BLOCK_TIMER("random step");
                auto& dec = decomps[0];
                // find a grow that reduces the cost
                for (int l = 0; l < 1; ++l)
                {
                    auto cpy = dec;
                    //                    RandomStepGrow(cpy.first);
                    RandomStepMerge(cpy.first);

                    cpy.second = cost(cpy.first.rectangles);
                    if (cpy.second < dec.second)
                    {
                        dec = cpy;
                        break;
                    }
                }
            }

            auto min_el =
                std::min_element(decomps.begin(), decomps.end(), [](auto& a, auto& b) { return a.second < b.second; });
            auto max_el =
                std::max_element(decomps.begin(), decomps.end(), [](auto& a, auto& b) { return a.second < b.second; });

            if (max_el->second > min_el->second + 1)
            {
                //                *max_el = *min_el;
            }


            if (min_el->second < current_best)
            {
                current_best          = min_el->second;
                not_improved_in_a_row = 0;
            }
            else
            {
                not_improved_in_a_row++;

                if (not_improved_in_a_row == converge_its)
                {
                    break;
                }
            }


            std::cout << "It " << it << " " << min_el->first << " C = " << min_el->second << std::endl;
        }
    }
    return Best().first;
}

void GrowAndShrinkDecomposition::RandomStepMerge(Decomposition& decomp)
{
    int ind = Random::uniformInt(0, decomp.rectangles.size() - 1);
    auto& r = decomp.rectangles[ind];

    std::vector<int> indices;
    for (int i = 0; i < decomp.rectangles.size(); ++i)
    {
        auto& r2 = decomp.rectangles[i];
        if (i != ind && r.Distance(r2) <= 2)
        {
            indices.push_back(i);
        }
    }

    if (indices.empty())
    {
        return;
    }

    auto& r2 = decomp.rectangles[indices[Random::uniformInt(0, indices.size() - 1)]];

    r  = Rect(r, r2);
    r2 = Rect();
    std::swap(decomp.rectangles.back(), decomp.rectangles[ind]);

    decomp = decomp.ShrinkIfPossible();
}

void GrowAndShrinkDecomposition::RandomStepGrow(Decomposition& decomp)
{
    int ind = Random::uniformInt(0, decomp.rectangles.size() - 1);

    int id = Random::uniformInt(0, 5);

    if (id < 3)
    {
        decomp.rectangles[ind].begin(id) -= 1;
    }
    else
    {
        decomp.rectangles[ind].end(id - 3) += 1;
    }

    // move to last spot
    std::swap(decomp.rectangles.back(), decomp.rectangles[ind]);

    decomp = decomp.ShrinkIfPossible();
}



}  // namespace RectangularDecomposition
}  // namespace Saiga
