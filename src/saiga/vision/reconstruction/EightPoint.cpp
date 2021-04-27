/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "EightPoint.h"

#include <random>

namespace Saiga
{
Mat3 NormalizePoints(const Vec2* src_points, Vec2* dst_points, int N)
{
    Vec2 center = Vec2::Zero();
    for (int i = 0; i < N; ++i)
    {
        center += src_points[i];
    }
    center /= N;


    double averageDistance = 0;
    for (int i = 0; i < N; ++i)
    {
        averageDistance += (src_points[i] - center).norm();
    }
    averageDistance /= N;

    double targetDistance = std::sqrt(2.0);
    double scale          = targetDistance / averageDistance;


    for (int i = 0; i < N; ++i)
    {
        dst_points[i] = (src_points[i] - center) * scale;
    }

    Mat3 T = Mat3::Identity();

    T(0, 2) = -center.x();
    T(1, 2) = -center.y();

    T       = T * scale;
    T(2, 2) = 1;
    return T;
}

Mat3 FundamentalMatrixEightPoint(const Vec2* points1, const Vec2* points2)
{
    using T = double;
    Eigen::Matrix<T, 8, 9, Eigen::RowMajor> A(8, 9);
    for (int i = 0; i < 8; ++i)
    {
        auto& p = *points1;
        auto& q = *points2;
        T px    = p(0);
        T py    = p(1);
        T qx    = q(0);
        T qy    = q(1);
        A.row(i) << px * qx, px * qy, px, py * qx, py * qy, py, qx, qy, 1;
        ++points1;
        ++points2;
    }

    Vec9 f;
    solveHomogeneousJacobiSVD(A, f);

    Mat3 F;
    F << f(0), f(3), f(6), f(1), f(4), f(7), f(2), f(5), f(8);
    F = enforceRank2(F);
    F = F * (1.0 / F(2, 2));
    return F;
}

Mat3 FundamentalMatrixEightPointNormalized(const Vec2* points1, const Vec2* points2)
{
    std::array<Vec2, 8> normalized_points1, normalized_points2;
    Mat3 T1 = NormalizePoints(points1, normalized_points1.data(), 8);
    Mat3 T2 = NormalizePoints(points2, normalized_points2.data(), 8);
    Mat3 F  = FundamentalMatrixEightPoint(normalized_points1.data(), normalized_points2.data());
    F       = T2.transpose() * F * T1;
    return F;
}

int EightPointRansac::solve(ArrayView<const Vec2> _points1, ArrayView<const Vec2> _points2, Mat3& bestE,
                            std::vector<int>& bestInlierMatches, std::vector<char>& inlierMask)
{
#pragma omp single
    {
        points1 = _points1;
        points2 = _points2;
        N       = points1.size();
    }



    int idx;
    idx = compute(points1.size());



#pragma omp single
    {
        //            std::cout << "best idx " << idx << std::endl;
        bestE = models[idx];


        bestInlierMatches.clear();
        bestInlierMatches.reserve(numInliers[idx]);
        for (int i = 0; i < N; ++i)
        {
            if (inliers[idx][i]) bestInlierMatches.push_back(i);
        }

        inlierMask = inliers[idx];
    }


    return numInliers[idx];
}

bool EightPointRansac::computeModel(const RansacBase::Subset& set, EightPointRansac::Model& model)
{
    std::array<Vec2, 8> A;
    std::array<Vec2, 8> B;

    for (auto i : Range(0, (int)set.size()))
    {
        A[i] = points1[set[i]];
        B[i] = points2[set[i]];
    }
    model = FundamentalMatrixEightPoint(A.data(), B.data());
    return true;
}

double EightPointRansac::computeResidual(const EightPointRansac::Model& model, int i)
{
    return EpipolarDistanceSquared(points1[i], points2[i], model);
}



}  // namespace Saiga
