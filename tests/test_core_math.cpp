/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#include "saiga/core/Core.h"
#include "saiga/core/math/HomogeneousLSE.h"
#include "saiga/core/math/all.h"
#include "saiga/core/util/Align.h"

#include "gtest/gtest.h"

using namespace Saiga;

TEST(Matrix, Print)
{
    Mat4 m = Mat4::Identity();
    std::cout << m << std::endl;
}

#ifndef SAIGA_USE_EIGEN
TEST(Matrix, View)
{
    std::vector<float> data(5, 2.0);
    for (auto d : data)
    {
        std::cout << d << std::endl;
    }


    Eigen::MatrixView<float, 5, 1, Eigen::ColMajor> view(data.data(), 1, 1);

    view(0, 0) = 3.f;

    for (auto d : data)
    {
        std::cout << d << std::endl;
    }

    EXPECT_EQ(data[0], 3.f);

    Vec5 cpy = view;
    EXPECT_EQ(cpy[0], 3.f);
    EXPECT_EQ(cpy[1], 2.f);

    EXPECT_EQ(view.at(0), 3.f);
    EXPECT_EQ(view.at(1), 2.f);
    Vec5 zerovec = Vec5::Zero();
    view         = zerovec;
    EXPECT_EQ(view.at(0), 0.f);
    EXPECT_EQ(view.at(1), 0.f);

    std::cout << std::endl;
    for (auto d : data)
    {
        std::cout << d << std::endl;
    }

    EXPECT_EQ(data[4], 0.f);
}
#endif
TEST(Matrix, inverse)
{
    Mat4 m  = Mat4::Identity() * 4;
    m(0,1) = 1;
    Mat4 m2 = m.inverse();

    std::cout << m2 << std::endl;
    Mat4 m3 = m2.inverse();
    std::cout << m3 << std::endl;
}

TEST(Matrix, RowCol)
{
    Mat4 m = Mat4::Identity();

    //    Vec4 v = m.col(0);
    //
    //    std::cout << v << std::endl;
}

// Regression: solveHomogeneousQR, solveHomogeneousCOD and solveHomogeneousBDCSVD picked the solution column with
// A.rows() instead of A.cols(), which is wrong (or out of bounds) for every non-square A.
TEST(HomogeneousLSE, NonSquare)
{
    using MatrixXd = Eigen::Matrix<double, -1, -1>;
    using VectorXd = Eigen::Matrix<double, -1, 1>;

    // Wide (3x4, full row rank) and tall (6x4, rank 3) matrices, both with a one-dimensional null space.
    MatrixXd wide(3, 4);
    for (int i = 0; i < wide.rows(); ++i)
    {
        for (int j = 0; j < wide.cols(); ++j)
        {
            wide(i, j) = std::sin(1.7 * i + 0.9 * j * j + 0.3);
        }
    }
    MatrixXd b(6, 3);
    for (int i = 0; i < b.rows(); ++i)
    {
        for (int j = 0; j < b.cols(); ++j)
        {
            b(i, j) = std::cos(0.8 * i * i + 1.1 * j + 0.2);
        }
    }
    MatrixXd tall = b * wide;

    for (const MatrixXd& A : {wide, tall})
    {
        auto check = [&](const VectorXd& x, const char* name)
        {
            ASSERT_EQ(x.rows(), A.cols()) << name;
            EXPECT_NEAR(x.norm(), 1, 1e-10) << name;
            EXPECT_LT((A * x).norm(), 1e-10) << name << " " << A.rows() << "x" << A.cols();
        };
        VectorXd x;
        solveHomogeneousLU(A, x);
        check(x, "LU");
        solveHomogeneousQR(A, x);
        check(x, "QR");
        solveHomogeneousCOD(A, x);
        check(x, "COD");
        solveHomogeneousJacobiSVD(A, x);
        check(x, "JacobiSVD");
        solveHomogeneousBDCSVD(A, x);
        check(x, "BDCSVD");
    }
}
