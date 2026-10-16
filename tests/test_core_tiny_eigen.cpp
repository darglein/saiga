/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "glog/logging.h"
#include "gtest/gtest.h"

#include "tiny-eigen/matrix.h"
#include "tiny-eigen/quaternion.h"
#include "tiny-eigen/sophus.h"

using namespace TinyEigen;

using Vec3f = Matrix<float, 3, 1>;
using Vec3d = Matrix<double, 3, 1>;
using Vec4f = Matrix<float, 4, 1>;
using Mat3f = Matrix<float, 3, 3>;
using Mat4d = Matrix<double, 4, 4>;
using Arr3i = Array<int, 3, 1, ColMajor>;
using Vec3i = Matrix<int, 3, 1>;

TEST(TinyEigen, All)
{
    EXPECT_TRUE((Vec3i(1, 2, 3).array() >= Vec3i(0, 2, 3).array()).all());
    EXPECT_FALSE((Vec3i(1, 2, 3).array() >= Vec3i(0, 3, 3).array()).all());
}

TEST(TinyEigen, ArrayData)
{
    Arr3i a = Vec3i(4, 5, 6).array();
    EXPECT_EQ(a.data()[0], 4);
    EXPECT_EQ(a.data()[2], 6);
}

TEST(TinyEigen, RowViewReduction)
{
    Mat3f m;
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            m(i, j) = float(i * 3 + j);
        }
    }

    // Row 1 is (3, 4, 5).
    EXPECT_FLOAT_EQ(m.row(1).sum(), 12);
    EXPECT_FLOAT_EQ(m.row(1).maxCoeff(), 5);
    EXPECT_FLOAT_EQ(m.row(1).squaredNorm(), 50);
    EXPECT_FLOAT_EQ((m.block<2, 2>(1, 1).sum()), 4 + 5 + 7 + 8);
}

TEST(TinyEigen, ConstTail)
{
    const Vec4f v(1, 2, 3, 4);
    Vec3f t = v.tail<3>();
    EXPECT_EQ(t, Vec3f(2, 3, 4));
}

TEST(TinyEigen, ScalarDividedByView)
{
    Vec4f v(1, 2, 4, 8);
    Matrix<float, 2, 1> r = 8.f / v.tail<2>();
    EXPECT_FLOAT_EQ(r(0), 2);
    EXPECT_FLOAT_EQ(r(1), 1);
}

TEST(TinyEigen, InverseWithPivotSwap)
{
    // The second pivot is zero, and the largest entry of that column is in an already processed row.
    Mat4d m = Mat4d::Zero();
    m(0, 0) = 1;
    m(0, 1) = 5;
    m(1, 2) = 1;
    m(2, 1) = 1;
    m(3, 3) = 1;

    Mat4d identity = m * m.inverse();
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            EXPECT_NEAR(identity(i, j), i == j ? 1 : 0, 1e-12) << i << " " << j;
        }
    }
}

TEST(TinyEigen, QuaternionFromOppositeVectors)
{
    Vec3f a(0, 0, 2);
    Vec3f b(0, 0, -1);
    auto q  = Quaternion<float>::FromTwoVectors(a, b);
    Vec3f r = q * a.normalized();
    EXPECT_TRUE(r.allFinite());
    EXPECT_NEAR((r - b).norm(), 0, 1e-6);
    EXPECT_NEAR(q.norm(), 1, 1e-6);
}

TEST(TinyEigen, QuaternionFromAngleAxisDouble)
{
    double angle = 1.0;
    auto q       = Quaternion<double>::FromAngleAxis(angle, Vec3d(0, 0, 1));
    EXPECT_DOUBLE_EQ(q.z(), std::sin(angle / 2));
    EXPECT_DOUBLE_EQ(q.w(), std::cos(angle / 2));
}

TEST(TinyEigen, SophusCastOnConst)
{
    const Sophus::SE3d pose(Quaternion<double>::Identity(), Vec3d(1, 2, 3));
    Sophus::SE3f posef = pose.cast<float>();
    EXPECT_EQ(posef.translation(), Vec3f(1, 2, 3));
}

TEST(TinyGlog, CheckWithElse)
{
    bool else_branch = false;
    if (false)
        CHECK(true);
    else
        else_branch = true;
    EXPECT_TRUE(else_branch);

    else_branch = false;
    if (false)
        CHECK_EQ(1, 1);
    else
        else_branch = true;
    EXPECT_TRUE(else_branch);
}

TEST(TinyGlog, CheckOperatorPrecedence)
{
    // Without parentheses this would expand to '1 & (2 == 0)'.
    CHECK_EQ(1 & 2, 0);
    CHECK_NE(1 | 2, 0);
}

TEST(TinyGlogDeathTest, CheckFails)
{
    EXPECT_DEATH(CHECK(false), "");
    EXPECT_DEATH(CHECK_EQ(1, 2), "");
}
