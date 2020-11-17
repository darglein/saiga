/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "RectangularDecomposition.h"

#include "saiga/core/math/random.h"
#include "saiga/core/time/all.h"

#include <iomanip>

namespace std
{
template <>
struct hash<Saiga::ivec3>
{
    std::size_t operator()(const Saiga::ivec3& k) const
    {
        using std::hash;
        using std::size_t;
        using std::string;

        // Compute individual hash values for first,
        // second and third and combine them using XOR
        // and bit shifting:

        return ((hash<int>()(k(0)) ^ (hash<int>()(k(1)) << 1)) >> 1) ^ (hash<int>()(k(2)) << 1);
    }
};

}  // namespace std

namespace Saiga
{
namespace RectangularDecomposition
{
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
    std::unordered_map<ivec3, int> map;


    auto add_to_map = [&](const Rect& r, int n) {
        for (int z = r.begin(2); z < r.end(2); ++z)
        {
            for (int y = r.begin(1); y < r.end(1); ++y)
            {
                for (int x = r.begin(0); x < r.end(0); ++x)
                {
                    map[ivec3(x, y, z)] += n;
                }
            }
        }
    };

    auto removeable = [&](const Rect& r) {
        for (int z = r.begin(2); z < r.end(2); ++z)
        {
            for (int y = r.begin(1); y < r.end(1); ++y)
            {
                for (int x = r.begin(0); x < r.end(0); ++x)
                {
                    if (map[ivec3(x, y, z)] <= 0) return false;
                }
            }
        }
        return true;
    };


    for (auto& r : rectangles)
    {
        add_to_map(r, 1);
    }


    auto cp = rectangles;
    //    std::sort(cp.begin(), cp.end(), [](auto a, auto b) { return a.Volume() < b.Volume(); });



    Decomposition result;

    for (auto& r : cp)
    {
        add_to_map(r, -1);
        if (removeable(r))
        {
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
    std::unordered_map<ivec3, int> map;


    auto add_to_map = [&](const Rect& r, int n) {
        for (int z = r.begin(2); z < r.end(2); ++z)
        {
            for (int y = r.begin(1); y < r.end(1); ++y)
            {
                for (int x = r.begin(0); x < r.end(0); ++x)
                {
                    map[ivec3(x, y, z)] += n;
                }
            }
        }
    };

    auto removeable = [&](const Rect& r) {
        for (int z = r.begin(2); z < r.end(2); ++z)
        {
            for (int y = r.begin(1); y < r.end(1); ++y)
            {
                for (int x = r.begin(0); x < r.end(0); ++x)
                {
                    if (map[ivec3(x, y, z)] <= 0) return false;
                }
            }
        }
        return true;
    };


    for (auto& r : rectangles)
    {
        add_to_map(r, 1);
    }


    auto cp = rectangles;
    //    std::sort(cp.begin(), cp.end(), [](auto a, auto b) { return a.Volume() < b.Volume(); });



    Decomposition result;

    for (auto& r : cp)
    {
        add_to_map(r, -1);

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

                if (removeable_rect.Volume() < r1.Volume() && removeable(r1))
                {
                    removeable_rect = r1;
                    keep_rect       = r2;
                }

                if (removeable_rect.Volume() < r2.Volume() && removeable(r2))
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
    Decomposition result;
    for (auto p : points)
    {
        result.rectangles.push_back(Rect(p));
    }
    return result;
}

Decomposition RowMergeDecomposition::Compute(ArrayView<const ivec3> points)
{
    int dim = 0;

    int s1 = (dim + 1) % 3;
    int s2 = (dim + 2) % 3;
    int s3 = (dim + 3) % 3;


    Decomposition result;
    if (points.empty()) return result;


    std::vector<ivec3> copy(points.begin(), points.end());
    std::sort(copy.begin(), copy.end(),
              [=](auto a, auto b) { return std::tie(a(s1), a(s2), a(s3)) < std::tie(b(s1), b(s2), b(s3)); });

    Rect current = points.front();

    for (int i = 1; i < points.size(); ++i)
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

std::vector<ivec3> RemoveDuplicates(ArrayView<const ivec3> points)
{
    std::vector<ivec3> result(points.begin(), points.end());
    std::sort(result.begin(), result.end(),
              [](auto a, auto b) { return std::tie(a(0), a(1), a(2)) < std::tie(b(0), b(1), b(2)); });
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
}

Decomposition GrowAndShrinkDecomposition::Compute(ArrayView<const ivec3> points)
{
    {
        // Initialization
        if (0)
        {
            RowMergeDecomposition rm;
            auto d = rm.Compute(points);
            decomps.push_back({d, Cost(d)});
        }


        TrivialRectangularDecomposition triv;
        for (int i = 0; i < N; ++i)
        {
            auto d = triv.Compute(points);
            decomps.push_back({d, Cost(d)});
        }

        current_best          = Best().second;
        not_improved_in_a_row = 0;
    }

    {
        int swap_it = its / 2;
        // SAIGA_BLOCK_TIMER();
        for (int it = 0; it < its; ++it)
        {
            if (it == swap_it)
            {
                cost_radius = 1;
                for (auto& d : decomps)
                {
                    d.second = Cost(d.first);
                }
                not_improved_in_a_row = 0;
                current_best          = Best().second;
            }
#pragma omp parallel for num_threads(decomps.size())
            for (int d = 0; d < decomps.size(); ++d)
            {
                auto& dec = decomps[d];
                // find a grow that reduces the cost
                for (int l = 0; l < 1; ++l)
                {
                    auto cpy = dec;
                    //                    RandomStepGrow(cpy.first);
                    RandomStepMerge(cpy.first);

                    cpy.second = Cost(cpy.first);
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

                if (not_improved_in_a_row == 100)
                {
                    if (it < swap_it)
                        it = swap_it - 1;
                    else
                        break;
                    not_improved_in_a_row = 0;
                }
            }


            //            std::cout << "It " << it << " " << min_el->first << " C = " << min_el->second << std::endl;
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
        if (i != ind && r.Distance(r2) <= cost_radius + 1)
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
