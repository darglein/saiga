/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Epipolar.h"

namespace Saiga
{
Mat3 EssentialMatrix(const SE3& a, const SE3& b)
{
    //    SE3 rel = a * b.inverse();
    //    SE3 rel = b * a.inverse();
    //    return rel.rotationMatrix() * skew(rel.translation());
    SE3 T  = b * a.inverse();
    Mat3 E = skew(T.translation().normalized()) * T.rotationMatrix();
    return NormalizeEpipolarMatrix(E);
}

Mat3 EssentialMatrix(const Mat3& F, const IntrinsicsPinholed& K1, const IntrinsicsPinholed& K2)
{
    Mat3 E = K2.matrix().transpose() * F * K1.matrix();
    return NormalizeEpipolarMatrix(E);
}

Mat3 FundamentalMatrix(const Mat3& E, const IntrinsicsPinholed& K1, const IntrinsicsPinholed& K2)
{
    Mat3 F = K2.inverse().matrix().transpose() * E * K1.inverse().matrix();
    return NormalizeEpipolarMatrix(F);
}

Mat3 NormalizeEpipolarMatrix(const Mat3& EorF)
{
    double l = EorF.norm();

    Mat3 result = EorF;

    if (l > 1e-10)
    {
        result *= (1.0 / l);
    }

    return result;
}

double EpipolarDistanceSquared(const Vec2& p1, const Vec2& p2, const Mat3& F)
{
    Vec3 np1(p1(0), p1(1), 1);
    Vec3 np2(p2(0), p2(1), 1);
    Vec3 l         = F * np1;
    double d       = np2.transpose() * l;
    double lengSqr = l(0) * l(0) + l(1) * l(1);
    double disSqr  = d * d / lengSqr;
    return disSqr;
}

void decomposeEssentialMatrix(const Mat3& E, Mat3& R1, Mat3& R2, Vec3& t1, Vec3& t2)
{
    auto svdE = E.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);

    Mat3 U  = svdE.matrixU();
    Mat3 VT = svdE.matrixV().transpose();

    Mat3 W  = Mat3::Zero();
    W(0, 1) = -1;
    W(1, 0) = 1;
    W(2, 2) = 1;

    R1 = U * W * VT;
    R2 = U * W.transpose() * VT;

    t1 = U.col(2);
    t1.normalize();
    t2 = -t1;
}

std::array<SE3, 4> decomposeEssentialMatrix2(Mat3& E)
{
    Mat3 R1, R2;
    Vec3 t1, t2;
    decomposeEssentialMatrix(E, R1, R2, t1, t2);

    // A negative determinant means that R contains a reflection. This is not rigid transformation!
    if (R1.determinant() < 0)
    {
        // scaling the essential matrix by -1 is allowed
        E = -E;
        decomposeEssentialMatrix(E, R1, R2, t1, t2);
        E = -E;
    }
    Quat q1(R1);
    Quat q2(R2);
    std::array<SE3, 4> possibilities = {SE3{q1, t1}, SE3{q2, t1}, SE3{q1, t2}, SE3{q2, t2}};
    return possibilities;
}

std::pair<SE3, int> getValidTransformationFromE(Mat3& E, const Vec2* points1, const Vec2* points2, int N)
{
    int bestT                        = 0;
    int bestCount                    = 0;
    std::array<SE3, 4> possibilities = decomposeEssentialMatrix2(E);


    for (int i = 0; i < 4; ++i)
    {
        auto T = possibilities[i];
        // Triangulate all points and count how many points are in front of both cameras
        // the transformation with the most valid points wins.
        int count = 0;
        for (int j = 0; j < N; ++j)
        {
            auto wp     = TriangulateHomogeneous<double, false>(SE3(), T, points1[j], points2[j]);
            Vec3 otherP = T * wp;


            //            double error1 = (Vec2(wp(0) / wp(2), wp(1) / wp(2)) - points1[j]).norm();
            //            double error2 = (Vec2(otherP(0) / otherP(2), otherP(1) / otherP(2)) - points2[j]).norm();

            //            std::cout << "rep e " << error1 << " " << error2 << std::endl;

            if (wp.z() > 0 && otherP.z() > 0)
            {
                count++;
            }
        }

        if (count > bestCount)
        {
            bestCount = count;
            bestT     = i;
        }
    }
    return {possibilities[bestT], bestCount};
}

std::pair<SE3, int> getValidTransformationFromE(Mat3& E, const Vec2* points1, const Vec2* points2,
                                                ArrayView<char> inlier_mask, int N, int num_threads)
{
    std::array<SE3, 4> possibilities = decomposeEssentialMatrix2(E);
    std::array<int, 4> counts;

#pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < 4; ++i)
    {
        auto T = possibilities[i];
        // Triangulate all points and count how many points are in front of both cameras
        // the transformation with the most valid points wins.
        int count = 0;
        for (int j = 0; j < N; ++j)
        {
            if (!inlier_mask[j]) continue;
            auto wp     = TriangulateHomogeneous<double, false>(SE3(), T, points1[j], points2[j]);
            Vec3 otherP = T * wp;
            if (wp.z() > 0 && otherP.z() > 0)
            {
                count++;
            }
        }
        counts[i] = count;
    }

    int idx = std::distance(counts.begin(), std::max_element(counts.begin(), counts.end()));
    return {possibilities[idx], counts[idx]};
    ;
}



}  // namespace Saiga
