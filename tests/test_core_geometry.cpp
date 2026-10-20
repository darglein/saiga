/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/geometry/aabb.h"
#include "saiga/core/geometry/bspline.h"

#include "gtest/gtest.h"

#include <sstream>

using namespace Saiga;

// Regression test: operator<< of AABB and Bspline wrote (parts of) the output to std::cout instead of the given stream.
TEST(Geometry, StreamOperatorsWriteToStream)
{
    std::stringstream strm;
    strm << AABB(vec3(0, 1, 2), vec3(3, 4, 5));
    EXPECT_NE(strm.str().find("AABB"), std::string::npos);

    std::stringstream strm2;
    Bspline<vec2, float> spline({vec2(0, 0), vec2(1, 0), vec2(1, 1), vec2(0, 1)}, 3);
    strm2 << spline;
    EXPECT_NE(strm2.str().find("knots:"), std::string::npos);
    EXPECT_NE(strm2.str().find("control points:"), std::string::npos);
}
