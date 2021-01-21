/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "RectilinearOptimization.h"

#include "saiga/core/math/random.h"
#include "saiga/core/time/all.h"

#include <iomanip>



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
    while (true)
    {
        int merged = 0;
#if 1
        DiscreteBVH bvh(rectangles);
        for (int i = 0; i < bvh.leaves.size(); ++i)
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
    SAIGA_BLOCK_TIMER();
    DiscreteBVH bvh(rectangles);

    std::vector<std::pair<int, int>> result;
    for (int i = 0; i < rectangles.size(); ++i)
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

void MergeNeighbors(RectangleList& rectangles, const Cost& cost)
{
    bool changed = true;
    while (changed)
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
            for (int i = 0; i < inters.size(); ++i)
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
            for (int i = 0; i < inters.size(); ++i)
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
                for (int i = 0; i < inters.size(); ++i)
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
}

std::vector<int> AllIntersectingRects(const RectangleList& rectangles, const Rect& r)
{
    std::vector<int> result;
    //        for (auto r2 : rectangles)
    for (int i = 0; i < rectangles.size(); ++i)
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


}  // namespace RectangularDecomposition
}  // namespace Saiga
