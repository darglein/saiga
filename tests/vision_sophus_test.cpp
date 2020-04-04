/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */



#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/util/Random.h"

#include "gtest/gtest.h"

#include "compare_numbers.h"
namespace Saiga
{
TEST(Sohpus, SE3_Point)
{
    SE3 a  = Random::randomSE3();
    Vec3 x = Vec3::Random();

    auto q = a.unit_quaternion();
    auto t = a.translation();

    Vec3 y1 = a * x;
    Vec3 y2 = q * x + t;

    ExpectCloseRelative(y1, y2, 1e-20);
}

TEST(Sohpus, SE3_SE3)
{
    SE3 a = Random::randomSE3();
    SE3 b = Random::randomSE3();

    auto a_q = a.unit_quaternion();
    auto a_t = a.translation();

    auto b_q = b.unit_quaternion();
    auto b_t = b.translation();

    SE3 y1 = a * b;

    Quat y_q = (a_q * b_q).normalized();
    Vec3 y_t = a_q * b_t + a_t;


    ExpectCloseRelative(y1.unit_quaternion().coeffs(), y_q.coeffs(), 1e-20);
    ExpectCloseRelative(y1.translation(), y_t, 1e-20);
}

TEST(Sohpus, Sim3_Point)
{
    Sim3 a = Random::randomSim3();
    Vec3 x = Vec3::Random();

    auto q = a.rxso3().quaternion().normalized();
    auto t = a.translation();
    auto s = a.scale();

    Vec3 y1 = a * x;
    Vec3 y2 = s * (q * x) + t;

    ExpectCloseRelative(y1, y2, 1e-10);
}

TEST(Sohpus, Sim3_Sim3)
{
    Sim3 a = Random::randomSim3();
    Sim3 b = Random::randomSim3();

    auto q_a = a.rxso3().quaternion().normalized();
    auto t_a = a.translation();
    auto s_a = a.scale();

    auto q_b = b.rxso3().quaternion().normalized();
    auto t_b = b.translation();
    auto s_b = b.scale();

    Sim3 y1    = a * b;
    Quat q_y   = (q_a * q_b).normalized();
    Vec3 t_y   = s_a * (q_a * t_b) + t_a;
    double s_y = s_a * s_b;

    ExpectCloseRelative(y1.rxso3().quaternion().normalized().coeffs(), q_y.coeffs(), 1e-10);
    ExpectCloseRelative(y1.translation(), t_y, 1e-10);
    ExpectCloseRelative(y1.scale(), s_y, 1e-10);
}

}  // namespace Saiga
