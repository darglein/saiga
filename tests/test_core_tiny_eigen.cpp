/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "glog/logging.h"
#include "gtest/gtest.h"

#include <cmath>

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

    EXPECT_TRUE((Vec3f(1, 2, 3).array() < Vec3f(2, 3, 4).array()).all());
    EXPECT_FALSE((Vec3f(1, 2, 3).array() < Vec3f(2, 2, 4).array()).all());
}

TEST(TinyEigen, Any)
{
    EXPECT_TRUE((Vec3i(1, 2, 3).array() < Vec3i(0, 0, 4).array()).any());
    EXPECT_FALSE((Vec3f(1, 2, 3).array() < Vec3f(1, 2, 3).array()).any());
}

TEST(TinyEigen, MatrixReduction)
{
    Mat3f m = Mat3f::Identity() * 2.f;
    EXPECT_FLOAT_EQ(m.sum(), 6);
    EXPECT_FLOAT_EQ(m.squaredNorm(), 12);
    EXPECT_FLOAT_EQ(m.maxCoeff(), 2);
    EXPECT_FLOAT_EQ(m.minCoeff(), 0);
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

TEST(TinyEigen, InverseRandom)
{
    // Compare A * inv(A) with the identity for a few fixed, well conditioned matrices of different sizes.
    Matrix<double, 5, 5> a;
    for (int i = 0; i < 5; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            a(i, j) = std::sin(1.3 * i + 0.7 * j * j) + (i == j ? 3 : 0);
        }
    }
    auto id5 = a * a.inverse();
    for (int i = 0; i < 5; ++i)
    {
        for (int j = 0; j < 5; ++j)
        {
            EXPECT_NEAR(id5(i, j), i == j ? 1 : 0, 1e-12) << i << " " << j;
        }
    }

    Mat3f b;
    b(0, 0)  = 2;
    b(0, 1)  = 1;
    b(0, 2)  = 0;
    b(1, 0)  = -1;
    b(1, 1)  = 3;
    b(1, 2)  = 1;
    b(2, 0)  = 0.5f;
    b(2, 1)  = 0;
    b(2, 2)  = 4;
    auto id3 = b * b.inverse();
    for (int i = 0; i < 3; ++i)
    {
        for (int j = 0; j < 3; ++j)
        {
            EXPECT_NEAR(id3(i, j), i == j ? 1 : 0, 1e-6) << i << " " << j;
        }
    }

    Matrix<double, 2, 2> c;
    c(0, 0) = 4;
    c(0, 1) = 7;
    c(1, 0) = 2;
    c(1, 1) = 6;
    auto ci = c.inverse();
    EXPECT_NEAR(ci(0, 0), 0.6, 1e-12);
    EXPECT_NEAR(ci(0, 1), -0.7, 1e-12);
    EXPECT_NEAR(ci(1, 0), -0.2, 1e-12);
    EXPECT_NEAR(ci(1, 1), 0.4, 1e-12);
}

TEST(TinyEigen, InverseOfSingularMatrixIsIdentity)
{
    Mat4d m = Mat4d::Zero();
    m(0, 0) = 1;
    m(1, 1) = 1;
    EXPECT_EQ(m.inverse(), Mat4d::Identity());
}

TEST(TinyEigen, InverseWithTinyPivot)
{
    // Regression: the 4x4 Gauss-Jordan inverse only swapped rows for an exactly zero pivot. With the tiny first
    // pivot 1e-8 the elimination cancelled the other entries in float and returned a badly wrong inverse.
    Matrix<float, 4, 4> m = Matrix<float, 4, 4>::Identity();
    m(0, 0)               = 1e-8f;
    m(0, 1)               = 1;
    m(1, 0)               = 1;
    m(1, 1)               = 1;

    Matrix<float, 4, 4> identity = m * m.inverse();
    for (int i = 0; i < 4; ++i)
    {
        for (int j = 0; j < 4; ++j)
        {
            EXPECT_NEAR(identity(i, j), i == j ? 1 : 0, 1e-5) << i << " " << j;
        }
    }
}

TEST(TinyEigen, QuaternionInverse)
{
    Quaternion<double> q(2, 0, 0, 0);
    auto inv = q.inverse();
    EXPECT_DOUBLE_EQ(inv.w(), 0.5);
    EXPECT_DOUBLE_EQ((q * inv).w(), 1);

    Quaternion<double> r(1, 2, 3, 4);
    auto id = r * r.inverse();
    EXPECT_NEAR(id.w(), 1, 1e-12);
    EXPECT_NEAR(id.vec().norm(), 0, 1e-12);

    EXPECT_EQ(r.conjugate().coeffs(), Quaternion<double>(1, -2, -3, -4).coeffs());
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

TEST(TinyGlog, CheckEvaluatesOnce)
{
    int calls = 0;
    auto f    = [&]() { return ++calls; };
    CHECK_EQ(f(), 1);
    EXPECT_EQ(calls, 1);
    CHECK(f() == 2);
    EXPECT_EQ(calls, 2);
}

TEST(TinyGlog, CheckNotNullReturnsValue)
{
    int value = 5;
    int* p    = &value;
    int* q    = CHECK_NOTNULL(p);
    EXPECT_EQ(q, p);
    EXPECT_EQ(*CHECK_NOTNULL(&value), 5);
}

TEST(TinyGlog, VlogAboveZeroIsSilent)
{
    int evaluated = 0;
    auto f        = [&]() { return ++evaluated; };
    VLOG(1) << f();
    EXPECT_EQ(evaluated, 0);
    VLOG(0) << f() << "\n";
    EXPECT_EQ(evaluated, 1);
}

TEST(TinyGlogDeathTest, CheckFails)
{
    EXPECT_DEATH(CHECK(false), "Check failed");
    EXPECT_DEATH(CHECK_EQ(1, 2) << "custom message", "\\(1 == 2\\) 1 == 2(.|\n)*custom message");
    int* null = nullptr;
    EXPECT_DEATH(CHECK_NOTNULL(null), "null != nullptr");
}
