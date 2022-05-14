/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#include "saiga/core/Core.h"
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
