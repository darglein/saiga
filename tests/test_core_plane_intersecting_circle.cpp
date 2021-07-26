/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "saiga/core/geometry/plane.h"
#include "saiga/core/geometry/sphere.h"

#include "gtest/gtest.h"

namespace Saiga
{
TEST(IntersectingCircle, noIntersection)
{
    Plane plane(vec3(-2.0f, 0.0f, 0.0f), vec3(1.0f, 0.0f, 0.0f));
    Sphere sphere(vec3(0.0f, 0.0f, 0.0f), 1.0f);

    {
        auto [point, radius] = plane.intersectingCircle(sphere.pos, sphere.r);

        ASSERT_NEAR(point.x(), -2.0f, 1e-5f);
        ASSERT_NEAR(point.y(), 0.0f, 1e-5f);
        ASSERT_NEAR(point.z(), 0.0f, 1e-5f);
        ASSERT_NEAR(radius, 0.0f, 1e-5f);
    }
    plane = plane.invert();

    {
        auto [point, radius] = plane.intersectingCircle(sphere.pos, sphere.r);

        ASSERT_NEAR(point.x(), -2.0f, 1e-5f);
        ASSERT_NEAR(point.y(), 0.0f, 1e-5f);
        ASSERT_NEAR(point.z(), 0.0f, 1e-5f);
        ASSERT_NEAR(radius, 0.0f, 1e-5f);
    }
}

TEST(IntersectingCircle, pointIntersection)
{
    Plane plane(vec3(1.0f, 0.0f, 0.0f), vec3(-1.0f, 0.0f, 0.0f));
    Sphere sphere(vec3(0.0f, 0.0f, 0.0f), 1.0f);

    {
        auto [point, radius] = plane.intersectingCircle(sphere.pos, sphere.r);

        ASSERT_NEAR(point.x(), 1.0f, 1e-5f);
        ASSERT_NEAR(point.y(), 0.0f, 1e-5f);
        ASSERT_NEAR(point.z(), 0.0f, 1e-5f);
        ASSERT_NEAR(radius, 0.0f, 1e-5f);
    }
    plane = plane.invert();

    {
        auto [point, radius] = plane.intersectingCircle(sphere.pos, sphere.r);

        ASSERT_NEAR(point.x(), 1.0f, 1e-5f);
        ASSERT_NEAR(point.y(), 0.0f, 1e-5f);
        ASSERT_NEAR(point.z(), 0.0f, 1e-5f);
        ASSERT_NEAR(radius, 0.0f, 1e-5f);
    }
}

TEST(IntersectingCircle, centerIntersection)
{
    Plane plane(vec3(0.0f, 0.0f, 0.0f), vec3(1.0f, 0.0f, 0.0f));
    Sphere sphere(vec3(0.0f, 0.0f, 0.0f), 1.0f);

    {
        auto [point, radius] = plane.intersectingCircle(sphere.pos, sphere.r);

        ASSERT_NEAR(point.x(), 0.0f, 1e-5f);
        ASSERT_NEAR(point.y(), 0.0f, 1e-5f);
        ASSERT_NEAR(point.z(), 0.0f, 1e-5f);
        ASSERT_NEAR(radius, 1.0f, 1e-5f);
    }
    plane = plane.invert();

    {
        auto [point, radius] = plane.intersectingCircle(sphere.pos, sphere.r);

        ASSERT_NEAR(point.x(), 0.0f, 1e-5f);
        ASSERT_NEAR(point.y(), 0.0f, 1e-5f);
        ASSERT_NEAR(point.z(), 0.0f, 1e-5f);
        ASSERT_NEAR(radius, 1.0f, 1e-5f);
    }
}

TEST(IntersectingCircle, halfwayIntersection)
{
    Plane plane(vec3(0.0f, 0.0f, 0.0f), vec3(1.0f, 0.0f, 0.0f));
    Sphere sphere(vec3(1.0f, 0.0f, 0.0f), 2.0f);

    {
        auto [point, radius] = plane.intersectingCircle(sphere.pos, sphere.r);

        ASSERT_NEAR(point.x(), 0.0f, 1e-5f);
        ASSERT_NEAR(point.y(), 0.0f, 1e-5f);
        ASSERT_NEAR(point.z(), 0.0f, 1e-5f);
        ASSERT_NEAR(radius, 1.732050808f, 1e-5f);
    }
    plane = plane.invert();

    {
        auto [point, radius] = plane.intersectingCircle(sphere.pos, sphere.r);

        ASSERT_NEAR(point.x(), 0.0f, 1e-5f);
        ASSERT_NEAR(point.y(), 0.0f, 1e-5f);
        ASSERT_NEAR(point.z(), 0.0f, 1e-5f);
        ASSERT_NEAR(radius, 1.732050808f, 1e-5f);
    }
}

TEST(IntersectingCircle, XYZplaneIntersection)
{
    Plane plane(vec3(0.5f, 0.5f, 0.25f), vec3(0.0f, 1.0f, 0.0f));
    Sphere sphere(vec3(1.0f, 1.0f, 0.0f), 8.0f);

    {
        auto [point, radius] = plane.intersectingCircle(sphere.pos, sphere.r);

        ASSERT_NEAR(point.x(), 1.0f, 1e-5f);
        ASSERT_NEAR(point.y(), 0.5f, 1e-5f);
        ASSERT_NEAR(point.z(), 0.0f, 1e-5f);
        ASSERT_NEAR(radius, 7.984359711f, 1e-5f);
    }

    plane = plane.invert();

    {
        auto [point, radius] = plane.intersectingCircle(sphere.pos, sphere.r);

        ASSERT_NEAR(point.x(), 1.0f, 1e-5f);
        ASSERT_NEAR(point.y(), 0.5f, 1e-5f);
        ASSERT_NEAR(point.z(), 0.0f, 1e-5f);
        ASSERT_NEAR(radius, 7.984359711f, 1e-5f);
    }
}

}  // namespace Saiga
