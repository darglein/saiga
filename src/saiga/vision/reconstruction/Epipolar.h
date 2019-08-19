/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionIncludes.h"
#include "saiga/vision/VisionTypes.h"

#include "Triangulation.h"


namespace Saiga
{
/**
 * Computes the essential matrix from given camera extrinsics.
 * The resulting essential matrix maps points of (a) to lines in (b).
 *
 * 1. Computes relative rotation and translation [R|t] between cameras
 * 2. The essential matrix E is then: E = R[t]x
 *
 *
 * Hartley - Chapter 9.6 Essential Matrix (page 257):
 *
 * Given two normalized cameras P, P' with
 * P  = [I | 0]
 * P' = [R | t]
 * then the essential matrix mapping a point x in P to a line l in P' is
 * E = [t]xR
 *
 * If P,P' are not normalized we multiply P^-1 right sided to both matrices.
 * P2  = P  * P^-1 = [I|0]
 * P2' = P2 * P^-1 = [R|t]
 *
 * Relation to fundamental Matrix:
 * E = K'^T*F*K
 * F = K'^T^-1 * E * K^-1
 *
 */
inline Mat3 EssentialMatrix(const SE3& a, const SE3& b)
{
    //    SE3 rel = a * b.inverse();
    //    SE3 rel = b * a.inverse();
    //    return rel.rotationMatrix() * skew(rel.translation());
    SE3 T  = b * a.inverse();
    Mat3 E = skew(T.translation()) * T.rotationMatrix();
    E *= 1.0 / E(2, 2);
    return E;
}


/**
 * Computes the Fundamental Matrix given an essential matrix and
 * the camera intrinsics.
 * Assumes a pinhole camera model!
 *
 * F = K'^T^-1 * E * K^-1
 * F = K2^-T * E * K1^-1
 */
inline Mat3 FundamentalMatrix(const Mat3& E, const Intrinsics4& K1, const Intrinsics4& K2)
{
    Mat3 F = K2.inverse().matrix().transpose() * E * K1.inverse().matrix();
    F *= 1.0 / F(2, 2);
    return F;
}

/**
 * Computes the squared distance of point 2 to the epipolar line of point 1.
 */
inline double EpipolarDistanceSquared(const Vec2& p1, const Vec2& p2, const Mat3& F)
{
    Vec3 np1(p1(0), p1(1), 1);
    Vec3 np2(p2(0), p2(1), 1);
    Vec3 l         = F * np1;
    double d       = np2.transpose() * l;
    double lengSqr = l(0) * l(0) + l(1) * l(1);
    double disSqr  = d * d / lengSqr;
    return disSqr;
}



// estimate the rotation and translation of the camera given the essential matrix E
// see:
// Richard Hartley and Andrew Zisserman (2003). Multiple View Geometry in computer vision
// http://isit.u-clermont1.fr/~ab/Classes/DIKU-3DCV2/Handouts/Lecture16.pdf
inline void decomposeEssentialMatrix(const Mat3& E, Mat3& R1, Mat3& R2, Vec3& t1, Vec3& t2)
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

inline std::array<SE3, 4> decomposeEssentialMatrix2(Mat3& E)
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
    }
    Quat q1(R1);
    Quat q2(R2);
    std::array<SE3, 4> possibilities = {SE3{q1, t1}, SE3{q2, t1}, SE3{q1, t2}, SE3{q2, t2}};
    return possibilities;
}

inline std::pair<SE3, int> getValidTransformationFromE(Mat3& E, Vec2* points1, Vec2* points2, int N)
{
    int bestT                        = 0;
    int bestCount                    = 0;
    std::array<SE3, 4> possibilities = decomposeEssentialMatrix2(E);

    Triangulation<double> triangulation;

    for (int i = 0; i < 4; ++i)
    {
        auto T = possibilities[i];
        // Triangulate all points and count how many points are in front of both cameras
        // the transformation with the most valid points wins.
        int count = 0;
        for (int j = 0; j < N; ++j)
        {
            auto wp     = triangulation.triangulateHomogeneous(SE3(), T, points1[j], points2[j]);
            Vec3 otherP = T * wp;


            //            double error1 = (Vec2(wp(0) / wp(2), wp(1) / wp(2)) - points1[j]).norm();
            //            double error2 = (Vec2(otherP(0) / otherP(2), otherP(1) / otherP(2)) - points2[j]).norm();

            //            std::cout << "rep e " << error1 << " " << error2 << std::endl;

            if (wp.z() > 0 && otherP.z() > 0)
            {
                count++;
            }
        }

        //        std::cout << "Conf " << i << " Infront: " << count << std::endl;
        if (count > bestCount)
        {
            bestCount = count;
            bestT     = i;
        }
    }
    return {possibilities[bestT], bestCount};
}

}  // namespace Saiga
