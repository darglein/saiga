/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "saiga/core/geometry/intersection.h"
#include "saiga/core/geometry/plane.h"
#include "saiga/core/geometry/sphere.h"

#include "gtest/gtest.h"

namespace Saiga
{
static bool intersectSAT(vec3 points[8], const Sphere& sphere)
{
    vec3 frustumCenter = points[0] + points[1] + points[2] + points[3] + points[4] + points[5] + points[6] + points[7];
    frustumCenter /= 8.0f;

    float max = -10000000.0f;

    const vec3 frustumCenterToSphereCenter = sphere.pos - frustumCenter;
    const vec3 axis                        = frustumCenterToSphereCenter.normalized();

    for (int i = 0; i < 8; ++i)
    {
        vec3& currentCorner = points[i];
        vec3 v              = currentCorner - frustumCenter;
        float proj          = dot(v, axis);

        if (max < proj) max = proj;
    }

    if (length(frustumCenterToSphereCenter) - max - sphere.r > 0 && length(frustumCenterToSphereCenter) > 0)
        return false;
    else
        return true;
}

class ClustererTest : public ::testing::Test
{
   protected:
    ClustererTest() {}

    ~ClustererTest() override {}

    void SetUp() override {}

    void TearDown() override {}

    void clusterLights()
    {
        clusterCache.resize(clusterCount);
        for (int c = 0; c < clusterCount; ++c)
        {
            clusterCache[c].clear();
            clusterCache[c].reserve(avgAllowedItemsPerCluster);
        }

        int itemCount = 0;

        for (int i = 0; i < clusterData.size(); ++i)
        {
            Sphere& sphere     = clusterData[i];
            vec3 sphereCenter  = sphere.pos;
            float sphereRadius = sphere.r;

            int x0 = 0, x1 = planesX.size() - 1;
            int y0 = 0, y1 = planesY.size() - 1;
            int z0 = 0, z1 = planesZ.size() - 1;

            int centerOutsideZ = 0;
            int centerOutsideY = 0;


            while (z0 < z1 && planesZ[z0].distance(sphereCenter) >= sphereRadius)
            {
                z0++;
            }
            if (--z0 < 0 && planesZ[0].distance(sphereCenter) < 0)
            {
                centerOutsideZ--;
            }
            z0 = std::max(0, z0);
            while (z1 >= z0 && -planesZ[z1].distance(sphereCenter) >= sphereRadius)
            {
                --z1;
            }
            if (++z1 > (int)planesZ.size() - 1 && planesZ[(int)planesZ.size() - 1].distance(sphereCenter) > 0)
            {
                centerOutsideZ++;
            }
            z1 = std::min(z1, (int)planesZ.size() - 1);
            if (z0 >= z1)
            {
                continue;
            }


            while (y0 < y1 && planesY[y0].distance(sphereCenter) >= sphereRadius)
            {
                y0++;
            }
            if (--y0 < 0 && planesY[0].distance(sphereCenter) < 0)
            {
                centerOutsideY--;
            }
            y0 = std::max(0, y0);
            while (y1 >= y0 && -planesY[y1].distance(sphereCenter) >= sphereRadius)
            {
                --y1;
            }
            if (++y1 > (int)planesY.size() - 1 && planesY[(int)planesY.size() - 1].distance(sphereCenter) > 0)
            {
                centerOutsideY++;
            }
            y1 = std::min(y1, (int)planesY.size() - 1);
            if (y0 >= y1)
            {
                continue;
            }


            while (x0 < x1 && planesX[x0].distance(sphereCenter) >= sphereRadius)
            {
                x0++;
            }
            --x0;
            x0 = std::max(0, x0);
            while (x1 >= x0 && -planesX[x1].distance(sphereCenter) >= sphereRadius)
            {
                --x1;
            }
            ++x1;
            x1 = std::min(x1, (int)planesX.size() - 1);
            if (x0 >= x1)
            {
                continue;
            }



            if (!refinement)
            {
                // This is without the sphere refinement
                for (int z = z0; z < z1; ++z)
                {
                    for (int y = y0; y < y1; ++y)
                    {
                        for (int x = x0; x < x1; ++x)
                        {
                            int tileIndex = x + clusterX * y + (clusterX * clusterY) * z;

                            clusterCache[tileIndex].push_back(i);
                            itemCount++;
                        }
                    }
                }
            }
            else
            {
                if (centerOutsideZ < 0)
                {
                    z0 = -(int)planesZ.size() * 2;
                }
                if (centerOutsideZ > 0)
                {
                    z1 = (int)planesZ.size() * 2;
                }
                int cz      = (z0 + z1);
                int centerZ = cz / 2;
                if (centerOutsideZ == 0 && cz % 2 == 0)
                {
                    float d0 = planesZ[z0].distance(sphereCenter);
                    float d1 = -planesZ[z1].distance(sphereCenter);
                    if (d0 <= d1) centerZ -= 1;
                }

                if (centerOutsideY < 0)
                {
                    y0 = -(int)planesY.size() * 2;
                }
                if (centerOutsideY > 0)
                {
                    y1 = (int)planesY.size() * 2;
                }
                int cy      = (y0 + y1);
                int centerY = cy / 2;
                if (centerOutsideY == 0 && cy % 2 == 0)
                {
                    float d0 = planesY[y0].distance(sphereCenter);
                    float d1 = -planesY[y1].distance(sphereCenter);
                    if (d0 <= d1) centerY -= 1;
                }

                Sphere lightSphere(sphereCenter, sphereRadius);

                z0 = std::max(0, z0);
                z1 = std::min(z1, (int)planesZ.size() - 1);
                y0 = std::max(0, y0);
                y1 = std::min(y1, (int)planesY.size() - 1);

                for (int z = z0; z < z1; ++z)
                {
                    Sphere zLight = lightSphere;
                    if (z != centerZ)
                    {
                        Plane plane          = (z < centerZ) ? planesZ[z + 1] : planesZ[z].invert();
                        auto [point, radius] = plane.intersectingCircle(zLight.pos, zLight.r);
                        zLight.pos           = point;
                        zLight.r             = radius;
                        if (zLight.r < 1e-5) continue;
                    }
                    for (int y = y0; y < y1; ++y)
                    {
                        Sphere yLight = zLight;
                        if (y != centerY)
                        {
                            Plane plane          = (y < centerY) ? planesY[y + 1] : planesY[y].invert();
                            auto [point, radius] = plane.intersectingCircle(yLight.pos, yLight.r);
                            yLight.pos           = point;
                            yLight.r             = radius;
                            if (yLight.r < 1e-5) continue;
                        }
                        int x = x0;
                        while (x < x1 && planesX[x].distance(yLight.pos) >= yLight.r) x++;
                        --x;
                        x      = std::max(x0, x);
                        int xs = x1;
                        while (xs >= x && -planesX[xs].distance(yLight.pos) >= yLight.r) --xs;
                        ++xs;
                        xs = std::min(xs, x1);

                        for (; x < xs; ++x)
                        {
                            int tileIndex = x + clusterX * y + (clusterX * clusterY) * z;

                            clusterCache[tileIndex].push_back(i);
                            itemCount++;
                        }
                    }
                }
            }
        }
    }

    void clusterLightsSAT()
    {
        clusterCacheSat.resize(clusterCount);
        for (int c = 0; c < clusterCount; ++c)
        {
            clusterCacheSat[c].clear();
            clusterCacheSat[c].reserve(avgAllowedItemsPerCluster);
        }

        int itemCount = 0;

        for (int i = 0; i < clusterData.size(); ++i)
        {
            Sphere& sphere = clusterData[i];

            for (int x = 0; x < clusterX; ++x)
            {
                for (int y = 0; y < clusterY; ++y)
                {
                    for (int z = 0; z < clusterZ; ++z)
                    {
                        int tileIndex = x + clusterX * y + (clusterX * clusterY) * z;

                        clusterPoints pts = points[tileIndex];

                        if (intersectSAT(&pts.nBL, sphere))
                        {
                            clusterCacheSat[tileIndex].push_back(i);
                            itemCount++;
                        }
                    }
                }
            }
        }
    }

    void buildClusters(int x_cl, int y_cl, int z_cl, float dx = 1, float dy = 1, float dz = 1)
    {
        clusterX     = x_cl;
        clusterY     = y_cl;
        clusterZ     = z_cl;
        clusterCount = x_cl * y_cl * z_cl;

        for (int x = 0; x <= x_cl; ++x)
        {
            Plane left(vec3(x * dx, 0.0f, 0.0f), vec3(1.0f, 0.0f, 0.0f));
            planesX.push_back(left);
        }

        for (int y = 0; y <= y_cl; ++y)
        {
            Plane bottom(vec3(0.0f, y * dy, 0.0f), vec3(0.0f, 1.0f, 0.0f));
            planesY.push_back(bottom);
        }

        for (int z = 0; z <= z_cl; ++z)
        {
            Plane near(vec3(0.0f, 0.0f, z * dz), vec3(0.0f, 0.0f, 1.0f));
            planesZ.push_back(near);
        }

        points.resize(clusterCount);
        for (int x = 0; x < x_cl; ++x)
        {
            for (int y = 0; y < y_cl; ++y)
            {
                for (int z = 0; z < z_cl; ++z)
                {
                    clusterPoints pts;
                    pts.nBL = vec3(x, y, z);
                    pts.nBR = vec3(x + 1, y, z);
                    pts.nTL = vec3(x, y + 1, z);
                    pts.nTR = vec3(x + 1, y + 1, z);
                    pts.fBL = vec3(x, y, z + 1);
                    pts.fBR = vec3(x + 1, y, z + 1);
                    pts.fTL = vec3(x, y + 1, z + 1);
                    pts.fTR = vec3(x + 1, y + 1, z + 1);

                    int tileIndex = x + clusterX * y + (clusterX * clusterY) * z;

                    points.at(tileIndex) = pts;
                }
            }
        }

        clusterData.clear();
    }

    void PrintCache(int cacheNum)
    {
        auto& cache = cacheNum == 1 ? clusterCacheSat : clusterCache;
        for (int z = 0; z < clusterZ; ++z)
        {
            Eigen::Matrix<int, -1, -1> mat(clusterY, clusterX);
            for (int y = 0; y < clusterY; ++y)
            {
                for (int x = 0; x < clusterX; ++x)
                {
                    mat(y, x) = cache[x + clusterX * y + (clusterX * clusterY) * z].size();
                }
            }
            std::cout << "Layer z = " << z << std::endl;
            std::cout << mat << std::endl;
        }
    }

    bool cacheCompare()
    {
        bool valid = true;
        for (int c = 0; c < clusterCount; ++c)
        {
            valid &= clusterCacheSat[c].size() == clusterCache[c].size();
            if (clusterCacheSat[c].size() != clusterCache[c].size())
            {
                std::cout << "Cluster " << c << " : is: " << clusterCache[c].size()
                          << " , expected: " << clusterCacheSat[c].size() << std::endl;
            }

            for (int i = 0; i < clusterCacheSat[c].size(); ++i)
            {
                valid &= clusterCacheSat[c][i] == clusterCache[c][i];
                if (clusterCacheSat[c][i] != clusterCache[c][i])
                {
                    std::cout << "Cluster " << c << " at " << i << " : is: " << clusterCache[c][i]
                              << " , expected: " << clusterCacheSat[c][i] << std::endl;
                }
            }
        }
        return valid;
    }

    std::vector<Plane> planesX;
    std::vector<Plane> planesY;
    std::vector<Plane> planesZ;
    std::vector<Sphere> clusterData;

    struct clusterPoints
    {
        vec3 nBL;
        vec3 nBR;
        vec3 nTL;
        vec3 nTR;

        vec3 fBL;
        vec3 fBR;
        vec3 fTL;
        vec3 fTR;
    };

    std::vector<clusterPoints> points;

    int clusterCount = 0;
    int clusterX     = 0;
    int clusterY     = 0;
    int clusterZ     = 0;
    bool refinement  = false;

    int avgAllowedItemsPerCluster = 128;
    std::vector<std::vector<int>> clusterCache;
    std::vector<std::vector<int>> clusterCacheSat;
};

TEST_F(ClustererTest, 2dXY)
{
    buildClusters(10, 10, 1, 1, 1, 0.001);
    Sphere s(vec3(5, 5, 0.0005f), 3.1);

    clusterData.push_back(s);

    refinement = true;
    clusterLights();
    clusterLightsSAT();

    PrintCache(0);
    PrintCache(1);
    ASSERT_EQ(cacheCompare(), true);
}

TEST_F(ClustererTest, sixPlanesOneSphereNoRefinement)
{
    buildClusters(1, 1, 1);

    Sphere sphereInside(vec3(0.5f, 0.5f, 0.5f), 0.25f);

    clusterData.push_back(sphereInside);

    clusterLights();

    ASSERT_EQ(clusterCache[0].size(), 1);  // one item
    ASSERT_EQ(clusterCache[0][0], 0);      // index 0

    Sphere sphereOutside(vec3(-10.0f, 0.0f, 0.0f), 0.5f);

    clusterData.clear();
    clusterData.push_back(sphereOutside);

    clusterLights();

    ASSERT_EQ(clusterCache[0].size(), 0);  // no item

    Sphere sphereIntersectingLowerEnd(vec3(0.0f, 0.0f, 0.0f), 0.5f);

    clusterData.clear();
    clusterData.push_back(sphereIntersectingLowerEnd);

    clusterLights();

    ASSERT_EQ(clusterCache[0].size(), 1);  // one item
    ASSERT_EQ(clusterCache[0][0], 0);      // index 0

    Sphere sphereIntersectingHigherEnd(vec3(1.0f, 1.0f, 1.0f), 0.5f);

    clusterData.clear();
    clusterData.push_back(sphereIntersectingHigherEnd);

    clusterLights();

    ASSERT_EQ(clusterCache[0].size(), 1);  // one item
    ASSERT_EQ(clusterCache[0][0], 0);      // index 0

    Sphere sphereAroundCluster(vec3(0.5f, 0.5f, 0.5f), 4.0f);

    clusterData.clear();
    clusterData.push_back(sphereAroundCluster);

    clusterLights();

    ASSERT_EQ(clusterCache[0].size(), 1);  // one item
    ASSERT_EQ(clusterCache[0][0], 0);      // index 0
}

TEST_F(ClustererTest, sixPlanesOneSphereWithRefinement)
{
    refinement = true;
    buildClusters(1, 1, 1);

    Sphere sphereInside(vec3(0.5f, 0.5f, 0.5f), 0.25f);

    clusterData.push_back(sphereInside);

    clusterLights();
    clusterLightsSAT();
    ASSERT_EQ(cacheCompare(), true);

    ASSERT_EQ(clusterCache[0].size(), 1);  // one item
    ASSERT_EQ(clusterCache[0][0], 0);      // index 0

    Sphere sphereOutside(vec3(-10.0f, 0.0f, 0.0f), 0.5f);

    clusterData.clear();
    clusterData.push_back(sphereOutside);

    clusterLights();
    clusterLightsSAT();
    ASSERT_EQ(cacheCompare(), true);

    ASSERT_EQ(clusterCache[0].size(), 0);  // no item

    Sphere sphereIntersecting(vec3(0.0f, 0.0f, 0.0f), 0.5f);

    clusterData.clear();
    clusterData.push_back(sphereIntersecting);

    clusterLights();
    clusterLightsSAT();
    ASSERT_EQ(cacheCompare(), true);

    ASSERT_EQ(clusterCache[0].size(), 1);  // one item
    ASSERT_EQ(clusterCache[0][0], 0);      // index 0

    Sphere sphereIntersectingHigherEnd(vec3(1.0f, 1.0f, 1.0f), 0.5f);

    clusterData.clear();
    clusterData.push_back(sphereIntersectingHigherEnd);

    clusterLights();
    clusterLightsSAT();
    ASSERT_EQ(cacheCompare(), true);

    ASSERT_EQ(clusterCache[0].size(), 1);  // one item
    ASSERT_EQ(clusterCache[0][0], 0);      // index 0

    Sphere sphereAroundCluster(vec3(0.5f, 0.5f, 0.5f), 4.0f);

    clusterData.clear();
    clusterData.push_back(sphereAroundCluster);

    clusterLights();
    clusterLightsSAT();
    ASSERT_EQ(cacheCompare(), true);

    ASSERT_EQ(clusterCache[0].size(), 1);  // one item
    ASSERT_EQ(clusterCache[0][0], 0);      // index 0
}

TEST_F(ClustererTest, ThreeCubedClustersThreeSpheresLowEnd)
{
    buildClusters(3, 3, 3);

    Sphere sphereOnEdge(vec3(0.0f, 0.0f, 0.0f), 2.5f);
    clusterData.push_back(sphereOnEdge);
    Sphere sphereCenterOutside(vec3(-0.1f, -0.1f, -0.1f), 2.7f);
    clusterData.push_back(sphereCenterOutside);
    Sphere sphereCenterInside(vec3(0.1f, 0.1f, 0.1f), 2.4f);
    clusterData.push_back(sphereCenterInside);

    clusterLights();

    for (int c = 0; c < clusterCount; ++c)
    {
        ASSERT_EQ(clusterCache[c].size(), 3);  // 3 items in each cluster
    }

    refinement = true;

    clusterLights();
    clusterLightsSAT();
    ASSERT_EQ(cacheCompare(), true);

    for (int z = 0; z < 3; ++z)
    {
        for (int y = 0; y < 3; ++y)
        {
            for (int x = 0; x < 3; ++x)
            {
                int tileIndex = x + clusterX * y + (clusterX * clusterY) * z;

                if ((x == 2 && y == 2) || (x == 2 && z == 2) || (y == 2 && z == 2))
                    ASSERT_EQ(clusterCache[tileIndex].size(), 0);  // no item in the cluster
                else
                    ASSERT_EQ(clusterCache[tileIndex].size(), 3);  // 3 items in the cluster
            }
        }
    }
}

TEST_F(ClustererTest, ThreeCubedClustersThreeSpheresHighEnd)
{
    buildClusters(3, 3, 3);

    Sphere sphereOnEdge(vec3(3.0f, 3.0f, 3.0f), 2.5f);
    clusterData.push_back(sphereOnEdge);
    Sphere sphereCenterOutside(vec3(3.1f, 3.1f, 3.1f), 2.7f);
    clusterData.push_back(sphereCenterOutside);
    Sphere sphereCenterInside(vec3(2.9f, 2.9f, 2.9f), 2.4f);
    clusterData.push_back(sphereCenterInside);

    clusterLights();

    for (int c = 0; c < clusterCount; ++c)
    {
        ASSERT_EQ(clusterCache[c].size(), 3);  // 3 items in each cluster
    }

    refinement = true;

    clusterLights();
    clusterLightsSAT();
    ASSERT_EQ(cacheCompare(), true);

    for (int z = 0; z < 3; ++z)
    {
        for (int y = 0; y < 3; ++y)
        {
            for (int x = 0; x < 3; ++x)
            {
                int tileIndex = x + clusterX * y + (clusterX * clusterY) * z;

                if ((x == 0 && y == 0) || (x == 0 && z == 0) || (y == 0 && z == 0))
                    ASSERT_EQ(clusterCache[tileIndex].size(), 0);  // no item in the cluster
                else
                    ASSERT_EQ(clusterCache[tileIndex].size(), 3);  // 3 items in the cluster
            }
        }
    }
}

TEST_F(ClustererTest, ThreeCubedClustersOneSphereCenter)
{
    buildClusters(3, 3, 3);

    Sphere sphereCenter(vec3(1.5f, 1.5f, 1.5f), 0.65f);
    clusterData.push_back(sphereCenter);

    clusterLights();

    for (int c = 0; c < clusterCount; ++c)
    {
        ASSERT_EQ(clusterCache[c].size(), 1);  // 1 item in each cluster
    }

    refinement = true;

    clusterLights();
    clusterLightsSAT();
    ASSERT_EQ(cacheCompare(), true);

    for (int z = 0; z < 3; ++z)
    {
        for (int y = 0; y < 3; ++y)
        {
            for (int x = 0; x < 3; ++x)
            {
                int tileIndex = x + clusterX * y + (clusterX * clusterY) * z;

                if ((x == 1 && y == 1) || (x == 1 && z == 1) || (y == 1 && z == 1))
                    ASSERT_EQ(clusterCache[tileIndex].size(), 1);  // 1 item in the cluster
                else
                    ASSERT_EQ(clusterCache[tileIndex].size(), 0);  // no item in the cluster
            }
        }
    }
}

TEST_F(ClustererTest, ThreeCubedClustersOneSphereCenterBigger)
{
    buildClusters(3, 3, 3);

    Sphere sphereCenter(vec3(1.5f, 1.5f, -1.5f), 10.0f);
    clusterData.push_back(sphereCenter);

    clusterLights();

    for (int c = 0; c < clusterCount; ++c)
    {
        ASSERT_EQ(clusterCache[c].size(), 1);  // 1 item in each cluster
    }

    refinement = true;

    clusterLights();
    clusterLightsSAT();
    ASSERT_EQ(cacheCompare(), true);

    for (int c = 0; c < clusterCount; ++c)
    {
        ASSERT_EQ(clusterCache[c].size(), 1);  // 1 item in each cluster
    }
}

}  // namespace Saiga
