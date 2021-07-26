/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#include "saiga/vision/cameraModel/MotionModel.h"
#include "saiga/vision/util/Random.h"

#include "gtest/gtest.h"

#include "compare_numbers.h"

using namespace Saiga;


void TestSE3(const SE3& a, const SE3& b)
{
    ExpectCloseRelative(a.matrix(), b.matrix(), 1e-10);
}


TEST(MotionModel, SimpleAccess)
{
    Random::setSeed(30467436);
    MotionModel::Settings settings;
    settings.valid_range = 2;
    MotionModel mm(settings);

    auto v = Random::randomSE3();

    mm.addRelativeMotion(v, 5);
    mm.addRelativeMotion(v, 8);
    mm.addRelativeMotion(v, 5000);


    // Direct access
    TestSE3(v, mm.predictVelocityForFrame(5).value());
    TestSE3(v, mm.predictVelocityForFrame(8).value());
    TestSE3(v, mm.predictVelocityForFrame(5000).value());

    // Easy extrapolation
    TestSE3(v, mm.predictVelocityForFrame(6).value());
    TestSE3(v, mm.predictVelocityForFrame(7).value());
    TestSE3(v, mm.predictVelocityForFrame(9).value());
    TestSE3(v, mm.predictVelocityForFrame(10).value());
    TestSE3(v, mm.predictVelocityForFrame(5001).value());
    TestSE3(v, mm.predictVelocityForFrame(5002).value());

    // Extrapolation too far
    EXPECT_FALSE(mm.predictVelocityForFrame(0).has_value());
    EXPECT_FALSE(mm.predictVelocityForFrame(11).has_value());
    EXPECT_FALSE(mm.predictVelocityForFrame(1245).has_value());
    EXPECT_FALSE(mm.predictVelocityForFrame(38457897).has_value());
}


TEST(MotionModel, Damping)
{
    MotionModel::Settings settings;
    settings.damping = 0.5;
    MotionModel mm(settings);


    Vec3 t(4, 2, 1);
    SE3 v(Quat::Identity(), t);

    mm.addRelativeMotion(v, 5);

    EXPECT_EQ(v, mm.predictVelocityForFrame(5));

    Vec3 t2 = mm.predictVelocityForFrame(6)->translation();
    EXPECT_EQ(t2, t * settings.damping);
}
