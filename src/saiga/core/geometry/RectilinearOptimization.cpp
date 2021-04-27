/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "RectilinearOptimization.h"

#include "saiga/core/math/random.h"
#include "saiga/core/time/all.h"

#include <iomanip>


constexpr bool verbose = false;

namespace Saiga
{
namespace RectangularDecomposition
{
void RemoveEmpty(RectangleList& rectangles)
{
    rectangles.erase(std::remove_if(rectangles.begin(), rectangles.end(), [](Rect& a) { return a.Empty(); }),
                     rectangles.end());
}


void MergeNeighborsSave(RectangleList& rectangles)
{
    if (rectangles.empty()) return;
    while (true)
    {
        int merged = 0;
#if 1
        DiscreteBVH bvh(rectangles);
        for (int i = 0; i < (int)bvh.leaves.size(); ++i)
        {
            auto& r1 = bvh.leaves[i];
            if (r1.Volume() == 0) continue;

            std::vector<int> result;
            bvh.DistanceIntersect(r1, 0, result);

            for (int j : result)
            {
                auto& r2 = bvh.leaves[j];
                if (i == j || r2.Volume() == 0) continue;

                if (r1.Distance(r2) == 0 && Rect(r1, r2).Volume() == r1.Volume() + r2.Volume())
                {
                    r1 = Rect(r1, r2);
                    r2.setZero();
                    merged++;
                }
            }
        }

#else


#    if 0
        for (int i = 0; i < result.rectangles.size(); ++i)
        {
            auto& r1 = result.rectangles[i];
            if (r1.Volume() == 0) continue;

            for (int j = i + 1; j < result.rectangles.size(); ++j)
            {
                auto& r2 = result.rectangles[j];
                if (r2.Volume() == 0) continue;

                if (r1.Distance(r2) == 0 && Rect(r1, r2).Volume() == r1.Volume() + r2.Volume())
                {
                    r1 = Rect(r1, r2);
                    r2.setZero();
                    merged++;
                }
            }
        }
#    else

        auto neighs = result.NeighborList(0);
        for (auto n : neighs)
        {
            auto& r1 = result.rectangles[n.first];
            if (r1.Volume() == 0) continue;
            auto& r2 = result.rectangles[n.second];
            if (r2.Volume() == 0) continue;

            if (r1.Distance(r2) == 0 && Rect(r1, r2).Volume() == r1.Volume() + r2.Volume())
            {
                r1 = Rect(r1, r2);
                r2.setZero();
                merged++;
            }
        }
#    endif
#endif

        //        std::cout << "MergeNeighborsSave " << merged << std::endl;
        if (merged <= 0)
        {
            break;
        }
        else
        {
            RemoveEmpty(rectangles);
        }
    }
}

std::vector<std::pair<int, int>> NeighborList(RectangleList& rectangles, int distance)
{
#if 1
    SAIGA_OPTIONAL_BLOCK_TIMER(verbose);
    DiscreteBVH bvh(rectangles);

    std::vector<std::pair<int, int>> result;
    for (int i = 0; i < (int)rectangles.size(); ++i)
    {
        auto& r1 = rectangles[i];
        if (r1.Volume() == 0) continue;

        std::vector<int> inds;
        bvh.DistanceIntersect(r1, distance, inds);

        for (int j : inds)
        {
            if (i >= j) continue;
            auto& r2 = rectangles[j];
            if (r2.Volume() == 0) continue;
            result.push_back({i, j});
        }
    }
#else


    std::vector<std::pair<int, int>> result;
    for (int i = 0; i < rectangles.size(); ++i)
    {
        auto& r1 = rectangles[i];


        if (r1.Volume() == 0) continue;
        for (int j = i + 1; j < rectangles.size(); ++j)
        {
            auto& r2 = rectangles[j];
            if (r2.Volume() == 0) continue;

            if (r1.Distance(r2) <= distance)
            {
                result.push_back({i, j});
            }
        }
    }
#endif
    return result;
}

int MergeNeighbors(RectangleList& rectangles, const Cost& cost, int max_iterations)
{
    if (rectangles.empty()) return 0;
    bool changed = true;
    int it       = 0;
    for (; it < max_iterations && changed; ++it)
    {
        changed     = false;
        auto neighs = NeighborList(rectangles, 1);
        for (auto n : neighs)
        {
            auto& r1 = rectangles[n.first];
            if (r1.Volume() == 0) continue;
            auto& r2 = rectangles[n.second];
            if (r2.Volume() == 0) continue;

            SAIGA_ASSERT(!r1.Intersect(r2));

            // 1. Let's compute the merged rectangle and then check later if this merge is viable
            Rect merged = Rect(r1, r2);

            // 2. Compute all intersecting rects towards the new merged Rectangle
            auto inters = AllIntersectingRects(rectangles, merged);
            std::vector<std::tuple<Rect, Rect, Rect>> shrunk(inters.size());
            bool found = false;

            // 3. Check if all intersecting rects can be shrunk to the merged rect.
            for (int i = 0; i < (int)inters.size(); ++i)
            {
                auto& r = rectangles[inters[i]];


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
            for (int i = 0; i < (int)inters.size(); ++i)
            {
                auto& r = rectangles[inters[i]];
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
                for (int i = 0; i < (int)inters.size(); ++i)
                {
                    auto& r = rectangles[inters[i]];

                    if (inters[i] == n.first)
                    {
                        r = merged;
                    }
                    else
                    {
                        r = std::get<1>(shrunk[i]);
                        rectangles.push_back(std::get<2>(shrunk[i]));
                    }
                }
                changed = true;
            }
        }
        if (changed)
        {
            RemoveEmpty(rectangles);
        }
    }
    return it;
}

std::vector<int> AllIntersectingRects(const RectangleList& rectangles, const Rect& r)
{
    std::vector<int> result;
    //        for (auto r2 : rectangles)
    for (int i = 0; i < (int)rectangles.size(); ++i)
    {
        if (r.Intersect(rectangles[i]))
        {
            result.push_back(i);
        }
    }
    return result;
}

void ShrinkIfPossible(RectangleList& rectangles)
{
    PointHashMap<3> map;
    for (auto& r : rectangles)
    {
        map.Add(r, 1);
    }

    RectangleList result;
    result.reserve(rectangles.size());

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
            result.push_back(keep_rect);
        }
    }
    rectangles = result;
}


static bool RandomStepMerge(RectangleList& rectangles)
{
    int ind = Random::uniformInt(0, rectangles.size() - 1);
    auto& r = rectangles[ind];

    std::vector<int> indices;
    for (int i = 0; i < (int)rectangles.size(); ++i)
    {
        auto& r2 = rectangles[i];
        if (i != ind && r.Distance(r2) <= 2)
        {
            indices.push_back(i);
        }
    }

    if (indices.empty())
    {
        return false;
    }

    auto& r2 = rectangles[indices[Random::uniformInt(0, indices.size() - 1)]];

    r  = Rect(r, r2);
    r2 = Rect();
    std::swap(rectangles.back(), rectangles[ind]);

    return true;
}

void MergeShrink(PointView points, RectangleList& rectangles, int its, int converge_its, const Cost& cost)
{
    if (rectangles.empty()) return;
    int not_improved_in_a_row = 0;
    RectangleList current_decomp;
    float current_cost;

    current_decomp        = rectangles;
    current_cost          = cost(current_decomp);
    not_improved_in_a_row = 0;

    {
        SAIGA_OPTIONAL_BLOCK_TIMER(verbose);
        for (int it = 0; it < its; ++it)
        {
            // find a grow that reduces the cost

            auto cpy      = current_decomp;
            auto new_cost = current_cost;
            //                    RandomStepGrow(cpy.first);
            if (RandomStepMerge(cpy))
            {
                //                ShrinkIfPossible(cpy);
                ShrinkIfPossible2(cpy, points);
                RemoveEmpty(cpy);
                new_cost = cost(cpy);
            }

            if (new_cost < current_cost)
            {
                current_decomp        = cpy;
                current_cost          = new_cost;
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

            if (verbose)
            {
                std::cout << "It " << it << " " << current_decomp.size() << " C = " << current_cost << std::endl;
            }
        }
    }
    rectangles = current_decomp;
}

static Rect Shrink(Rect rect, PointView points)
{
    // std::cout << "Shring " << rect << " " << points.size() << std::endl;
    if (points.empty())
    {
        return Rect();
    }


    Rect result = Rect(points.front());

    for (int i = 1; i < (int)points.size(); ++i)
    {
        result = Rect(result, Rect(points[i]));
    }
    return result;
}
void ShrinkIfPossible2(RectangleList& rectangles, PointView points)
{
    DiscreteBVH bvh(rectangles);

    std::vector<int> rect_ids(points.size());
    for (int i = 0; i < (int)points.size(); ++i)
    {
        std::vector<int> inds;
        bvh.DistanceIntersect(Rect(points[i]), -1, inds);
        SAIGA_ASSERT(!inds.empty());
        int idx     = *std::min_element(inds.begin(), inds.end());
        rect_ids[i] = idx;
    }

    std::vector<std::vector<ivec3>> points_per_rect(rectangles.size());
    for (int i = 0; i < (int)points.size(); ++i)
    {
        points_per_rect[rect_ids[i]].push_back(points[i]);
    }

    for (int i = 0; i < (int)rectangles.size(); ++i)
    {
        rectangles[i] = Shrink(rectangles[i], points_per_rect[i]);
    }
}


}  // namespace RectangularDecomposition
}  // namespace Saiga
