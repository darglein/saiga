/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */



#include "saiga/vision/VisionTypes.h"
#include "saiga/core/sophus/decoupled_sim3.h"
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

    ExpectCloseRelative(y1, y2, 1e-10);
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


    ExpectCloseRelative(y1.unit_quaternion().coeffs(), y_q.coeffs(), 1e-10);
    ExpectCloseRelative(y1.translation(), y_t, 1e-10);
}

TEST(Sohpus, Sim3_Identity)
{
    Sophus::DSim3<double> da;

    auto q            = Eigen::Quaternion<double>::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    auto s            = 1.0;

    ExpectCloseRelative(da.se3().unit_quaternion().coeffs(), q.coeffs(), 1e-10);
    ExpectCloseRelative(da.se3().translation(), t, 1e-10);
    ExpectCloseRelative(da.scale(), s, 1e-10);
}

TEST(Sohpus, Sim3_Point)
{
    Sophus::Sim3d a = Random::randomSim3();
    Sophus::DSim3<double> da(a);
    Vec3 x = Vec3::Random();

    auto q = a.rxso3().quaternion().normalized();
    auto t = a.translation();
    auto s = a.scale();

    Vec3 y1 = a * x;
    Vec3 y2 = s * (q * x) + t;
    Vec3 y3 = da * x;

    ExpectCloseRelative(y1, y2, 1e-10);
    ExpectCloseRelative(y3, y2, 1e-10);
}

TEST(Sohpus, Sim3_Inverse)
{
    Sophus::Sim3d a = Random::randomSim3();
    Sophus::DSim3<double> da(a);

    auto q = a.rxso3().quaternion().normalized();
    auto t = a.translation();
    auto s = a.scale();

    Sophus::Sim3d y1 = a.inverse();
    Quat q_y         = q.inverse();
    double s_y       = 1.0 / s;
    Vec3 t_y         = -s_y * (q_y * t);

    auto y3 = da.inverse();

    ExpectCloseRelative(y1.rxso3().quaternion().normalized().coeffs(), q_y.coeffs(), 1e-10);
    ExpectCloseRelative(y1.translation(), t_y, 1e-10);
    ExpectCloseRelative(y1.scale(), s_y, 1e-10);

    ExpectCloseRelative(y3.se3().unit_quaternion().coeffs(), q_y.coeffs(), 1e-10);
    ExpectCloseRelative(y3.se3().translation(), t_y, 1e-10);
    ExpectCloseRelative(y3.scale(), s_y, 1e-10);


    auto I = da * da.inverse();
    Sophus::DSim3<double> db;

    ExpectCloseRelative(I.se3().unit_quaternion().coeffs(), db.se3().unit_quaternion().coeffs(), 1e-10);
    ExpectCloseRelative(I.se3().translation(), db.se3().translation(), 1e-10);
    ExpectCloseRelative(I.scale(), db.scale(), 1e-10);
}

TEST(Sohpus, Sim3_Sim3)
{
    Sophus::Sim3d a = Random::randomSim3();
    Sophus::Sim3d b = Random::randomSim3();

    Sophus::DSim3<double> da(a);
    Sophus::DSim3<double> db(b);

    auto q_a = a.rxso3().quaternion().normalized();
    auto t_a = a.translation();
    auto s_a = a.scale();

    auto q_b = b.rxso3().quaternion().normalized();
    auto t_b = b.translation();
    auto s_b = b.scale();

    Sophus::Sim3d y1 = a * b;
    Quat q_y         = (q_a * q_b).normalized();
    Vec3 t_y         = s_a * (q_a * t_b) + t_a;
    double s_y       = s_a * s_b;

    auto y3 = da * db;

    ExpectCloseRelative(y1.rxso3().quaternion().normalized().coeffs(), q_y.coeffs(), 1e-10);
    ExpectCloseRelative(y1.translation(), t_y, 1e-10);
    ExpectCloseRelative(y1.scale(), s_y, 1e-10);

    ExpectCloseRelative(y3.se3().unit_quaternion().coeffs(), q_y.coeffs(), 1e-10);
    ExpectCloseRelative(y3.se3().translation(), t_y, 1e-10);
    ExpectCloseRelative(y3.scale(), s_y, 1e-10);
}

TEST(Sohpus, Map)
{
    std::array<double, 8> data;
    std::fill(data.begin(), data.end(), 0.0);


    //    Eigen::Map<DSim3> dsim3_data(data.data());
}


}  // namespace Saiga
