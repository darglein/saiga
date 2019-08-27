/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/Range.h"
#include "saiga/core/util/Thread/omp.h"
#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/kernels/BAPose.h"
#include "saiga/vision/kernels/Robust.h"

#include "p3p.h"

#include <vector>

namespace Saiga
{
/**
 * Given a frame with 2D to 3D matches and an initial guess, this class computes
 * the best pose.
 */
template <typename T>
class PNP
{
   public:
    using SE3Type = Sophus::SE3<T>;
    using Vec2    = Eigen::Matrix<T, 2, 1>;
    using Vec3    = Eigen::Matrix<T, 3, 1>;

    int solvePNPRansac(const AlignedVector<Vec3>& wps, const AlignedVector<Vec2>& ips,
                       std::vector<int>& bestInlierMatches, SE3Type& result)
    {
        int maxIterations        = 1000;
        constexpr int sampleSize = 4;
        double threshold         = 4.0 / 500;
        int N                    = wps.size();

        std::uniform_int_distribution<unsigned int> dis(0, N - 1);
        std::mt19937 gen = std::mt19937(92730469346UL);

        std::array<Vec3, sampleSize> A;
        std::array<Vec2, sampleSize> B;

        int bestInliers = 0;

        p3p p3psolver;
        for (int i = 0; i < maxIterations; ++i)
        {
            for (auto j : Range(0, sampleSize))
            {
                auto idx = dis(gen);
                A[j]     = wps[idx];
                B[j]     = ips[idx];
            }

            //            SE3 trans = extractSE3(dlt(A.data(), B.data(), N));
            SE3 trans;
            auto valid = p3psolver.solve(A.data(), B.data(), trans);
            if (!valid) continue;

            // just do a single iteration gauss newton on the 4 sample points
            trans = refinePose(trans, A.data(), B.data(), sampleSize, 1);

            std::vector<int> inlierMatches;
            int numInliers = 0;
            for (int j = 0; j < N; ++j)
            {
                auto ip2 = trans * wps[j];
                ip2      = ip2 / ip2.z();

                double error = (ip2.template segment<2>(0) - ips[j]).squaredNorm();
                //                std::cout << error << std::endl;

                if (error < threshold)
                {
                    inlierMatches.push_back(i);
                    numInliers++;
                }
            }


            if (numInliers > bestInliers)
            {
                bestInliers       = numInliers;
                result            = trans;
                bestInlierMatches = inlierMatches;
            }
        }

#if 0
        // refine again with all inliers

        AlignedVector<Vec3> wpIn;
        AlignedVector<Vec2> ipIn;
        for (auto i : bestInlierMatches)
        {
            wpIn.push_back(wps[i]);
            ipIn.push_back(ips[i]);
        }

        result = refinePose(result, wpIn.data(), ipIn.data(), bestInliers, 3);
#endif
        return bestInliers;
    }


    SE3 refinePose(const SE3& pose, const Vec3* worldPoints, const Vec2* normalizedImagePoints, int N, int iterations)
    {
        using MonoKernel = typename Saiga::Kernel::BAPoseMono<T, false, true>;
        using JType      = Eigen::Matrix<T, 6, 6>;
        using BType      = Eigen::Matrix<T, 6, 1>;
        typename MonoKernel::CameraType dummy;

        typename MonoKernel::JacobiType JrowM;
        JType JtJ;
        BType Jtb;

        SE3 guess = pose;
        for (auto it : Range(0, iterations))
        {
            JtJ.setZero();
            Jtb.setZero();
            double chi2sum = 0;

            for (auto i : Range(0, N))
            {
                auto&& wp = worldPoints[i];
                auto&& ip = normalizedImagePoints[i];

                Vec2 res;
                MonoKernel::evaluateResidualAndJacobian(dummy, guess, wp, ip, res, JrowM, 1);
                auto c2 = res.squaredNorm();

                chi2sum += c2;
                JtJ += (JrowM.transpose() * JrowM);
                Jtb += JrowM.transpose() * res;
            }

            BType x = JtJ.ldlt().solve(Jtb);
            guess   = SE3Type::exp(x) * guess;

            //            std::cout << "chi2 " << chi2sum << std::endl;
        }
        return guess;
    }


    Eigen::Matrix<T, 3, 4> dlt(Vec3* worldPoints, Vec2* normalizedImagePoints, int N)
    {
        Eigen::MatrixXd L(N * 2, 12);
        for (int i = 0; i < N; ++i)
        {
            auto wp = worldPoints[i];
            auto ip = normalizedImagePoints[i];
            auto nx = -ip.x();
            auto ny = -ip.y();
            Eigen::Matrix<double, 1, 12> l, l2;
            l << wp.x(), wp.y(), wp.z(), 1, 0, 0, 0, 0, nx * wp.x(), nx * wp.y(), nx * wp.z(), nx;
            l2 << 0, 0, 0, 0, wp.x(), wp.y(), wp.z(), 1, ny * wp.x(), ny * wp.y(), ny * wp.z(), ny;
            L.row(2 * i)     = l;
            L.row(2 * i + 1) = l2;
        }
        Eigen::JacobiSVD<Eigen::MatrixXd> svd = L.jacobiSvd(Eigen::ComputeFullU | Eigen::ComputeFullV);
        Eigen::MatrixXd V                     = svd.matrixV();
        auto p                                = V.col(11);
        Eigen::Matrix<T, 3, 4> P;
        P << p(0), p(1), p(2), p(3), p(4), p(5), p(6), p(7), p(8), p(9), p(10), p(11);
        return P;
    }

    SE3 extractSE3(const Eigen::Matrix<T, 3, 4>& P2)
    {
        Eigen::Matrix<T, 3, 4> P = P2;
        // P is a scaled version of the actual camera matrix.
        // we try to find R that is an orthonormal matrix
        Mat3 R = P.template block<3, 3>(0, 0);
        if (R.determinant() < 0)
        {
            P = -1.0 * P;
            R = -1.0 * R;
        }

        Vec3 t  = P.template block<3, 1>(0, 3);
        auto sc = R.norm();

        Quat q(R);


        Mat3 R2 = q.matrix();
        // also scale the translation by the amount the rotation was scaled
        t = t * R2.norm() / sc;


        SE3 result(q, t);

        return result;
    }
};


}  // namespace Saiga
