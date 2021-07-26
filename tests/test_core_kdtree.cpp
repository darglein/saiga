/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#include "saiga/core/Core.h"
#include "saiga/core/geometry/kdtree.h"
#include "saiga/core/math/all.h"
#include "saiga/core/util/Align.h"

#include "gtest/gtest.h"

using namespace Saiga;


using KDT = KDTree<3, vec3>;

std::vector<vec3> RandomPoints(int n)
{
    std::vector<vec3> result;
    for (int i = 0; i < n; ++i)
    {
        result.push_back(Random::MatrixUniform<vec3>(-1, 1));
    }
    return result;
}

std::vector<std::pair<float, int>> ComputeDistance(const std::vector<vec3>& points, const vec3& search_point)
{
    std::vector<std::pair<float, int>> result;

    for (size_t i = 0; i < points.size(); ++i)
    {
        float d2 = (points[i] - search_point).squaredNorm();
        result.push_back({d2, i});
    }
    std::sort(result.begin(), result.end());
    return result;
}

int NearestNeighborBruteForce(const std::vector<vec3>& points, const vec3& search_point)
{
    auto dis = ComputeDistance(points, search_point);
    return dis.front().second;
}

std::vector<int> KNearestNeighborBruteForce(const std::vector<vec3>& points, const vec3& search_point, int k)
{
    auto dis = ComputeDistance(points, search_point);

    std::vector<int> result;
    for (int i = 0; i < k; ++i)
    {
        result.push_back(dis[i].second);
    }
    return result;
}



std::vector<int> RadiusSearch(const std::vector<vec3>& points, const vec3& search_point, float r)
{
    std::vector<int> result;

    float r2 = r * r;

    for (size_t i = 0; i < points.size(); ++i)
    {
        float d2 = (points[i] - search_point).squaredNorm();
        if (d2 < r2)
        {
            result.push_back(i);
        }
    }

    return result;
}


TEST(kdtree, NearestNeighbour)
{
    Random::setSeed(30947643);
    auto points        = RandomPoints(1000);
    auto search_points = RandomPoints(10);
    KDT tree(points);

    for (auto sp : search_points)
    {
        EXPECT_EQ(NearestNeighborBruteForce(points, sp), tree.NearestNeighborSearch(sp));
    }
}


TEST(kdtree, KNearestNeighbour)
{
    Random::setSeed(30947643);
    auto points        = RandomPoints(1000);
    auto search_points = RandomPoints(10);
    int k              = 10;
    KDT tree(points);

    for (auto sp : search_points)
    {
        EXPECT_EQ(KNearestNeighborBruteForce(points, sp, k), tree.KNearestNeighborSearch(sp, k));
    }
}

TEST(kdtree, RadiusSearch)
{
    Random::setSeed(30947643);
    auto points        = RandomPoints(1000);
    auto search_points = RandomPoints(10);
    float r            = 0.3;
    KDT tree(points);

    for (auto sp : search_points)
    {
        EXPECT_EQ(RadiusSearch(points, sp, r), tree.RadiusSearch(sp, r));
    }
}
