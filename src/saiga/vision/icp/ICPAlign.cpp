/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "ICPAlign.h"

#include "saiga/core/util/assert.h"

namespace Saiga
{
namespace ICP
{
SE3 pointToPointIterative(const AlignedVector<Correspondence>& corrs, const SE3& guess, int innerIterations)
{
    SAIGA_ASSERT(corrs.size() >= 6);

    SE3 T = guess;
    Eigen::Matrix<double, 6, 6> JtJ;
    Eigen::Matrix<double, 6, 1> Jtb;

    for (int k = 0; k < innerIterations; ++k)
    {
        JtJ.setZero();
        Jtb.setZero();

        for (size_t i = 0; i < corrs.size(); ++i)
        {
            auto& corr = corrs[i];

            Vec3 sp = T * corr.srcPoint;


            Eigen::Matrix<double, 3, 6> Jrow;
            Jrow.block<3, 3>(0, 0) = Mat3::Identity();
            Jrow.block<3, 3>(0, 3) = -skew(sp);


            Vec3 res = corr.refPoint - sp;

            // use weight
            Jrow *= corr.weight;
            res *= corr.weight;

            JtJ += Jrow.transpose() * Jrow;
            Jtb += Jrow.transpose() * res;
        }
        Eigen::Matrix<double, 6, 1> x = JtJ.ldlt().solve(Jtb);
        T                             = SE3::exp(x) * T;
    }
    return T;
}

inline Quat orientationFromMixedMatrixUQ(const Mat3& M)
{
    // Closed-form solution of absolute orientation using unit quaternions
    // https://pdfs.semanticscholar.org/3120/a0e44d325c477397afcf94ea7f285a29684a.pdf

    // Upper triangle
    Mat4 N;
    N(0, 0) = M(0, 0) + M(1, 1) + M(2, 2);
    N(0, 1) = M(1, 2) - M(2, 1);
    N(0, 2) = M(2, 0) - M(0, 2);
    N(0, 3) = M(0, 1) - M(1, 0);
    N(1, 1) = M(0, 0) - M(1, 1) - M(2, 2);
    N(1, 2) = M(0, 1) + M(1, 0);
    N(1, 3) = M(2, 0) + M(0, 2);
    N(2, 2) = -M(0, 0) + M(1, 1) - M(2, 2);
    N(2, 3) = M(1, 2) + M(2, 1);
    N(3, 3) = -M(0, 0) - M(1, 1) + M(2, 2);
    N       = N.selfadjointView<Eigen::Upper>();

    Eigen::EigenSolver<Mat4> eigenSolver(N, true);

    Vec4 E             = eigenSolver.eigenvectors().col(0).real();
    auto largestRealEv = eigenSolver.eigenvalues()(0).real();
    for (auto i = 1; i < 4; ++i)
    {
        auto ev = eigenSolver.eigenvalues()(i).real();
        if (ev > largestRealEv)
        {
            E             = eigenSolver.eigenvectors().col(i).real();
            largestRealEv = ev;
        }
    }
    Quat R(E(0), E(1), E(2), E(3));
    return R.conjugate();
}

inline Quat orientationFromMixedMatrixSVD(const Mat3& M)
{
    // polar decomp
    // M = USV^T
    // R = UV^T
    Eigen::JacobiSVD svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Mat3 R = svd.matrixU() * svd.matrixV().transpose();
    return Quat(R);
}

SE3 pointToPointDirect(const AlignedVector<Correspondence>& corrs, const SE3& guess, int innerIterations)
{
    SE3 T = guess;
    for (int i = 0; i < innerIterations; ++i)
    {
        Vec3 meanRef(0, 0, 0);
        Vec3 meanSrc(0, 0, 0);

        for (auto c : corrs)
        {
            meanRef += c.refPoint;
            meanSrc += T * c.srcPoint;
        }
        meanRef /= corrs.size();
        meanSrc /= corrs.size();
        Vec3 t = meanRef - meanSrc;

        Mat3 M;
        M.setZero();
        for (auto c : corrs)
        {
            M += (c.refPoint - meanRef) * (T * c.srcPoint - meanSrc).transpose();
        }


        Quat R;

        if (1)
            R = orientationFromMixedMatrixUQ(M);
        else
            R = orientationFromMixedMatrixSVD(M);

        t = meanRef - R.conjugate() * meanSrc;
        T = SE3(Quat(R), t) * T;
    }
    return T;
}

SE3 pointToPlane(const AlignedVector<Correspondence>& corrs, const SE3& ref, const SE3& _src, int innerIterations)
{
    SAIGA_ASSERT(corrs.size() >= 6);
    auto src = _src;
    Eigen::Matrix<double, 6, 6> JtJ;
    Eigen::Matrix<double, 6, 1> Jtb;


    for (int k = 0; k < innerIterations; ++k)
    {
        // Make use of symmetry
        JtJ.triangularView<Eigen::Upper>().setZero();
        Jtb.setZero();

        for (size_t i = 0; i < corrs.size(); ++i)
        {
            auto& corr = corrs[i];

            Vec3 rp = ref * corr.refPoint;
            Vec3 rn = ref.so3() * corr.refNormal;
            Vec3 sp = src * corr.srcPoint;

            Eigen::Matrix<double, 6, 1> row;
            row.head<3>() = rn;
            // This is actually equal to:
            //      row.tail<3>() = -skew(sp).transpose() * rn;
            row.tail<3>() = sp.cross(rn);
            Vec3 di       = rp - sp;
            double res    = rn.dot(di);

            // use weight
            row *= corr.weight;
            res *= corr.weight;

            //            JtJ += row * row.transpose();
            JtJ += (row * row.transpose()).triangularView<Eigen::Upper>();
            Jtb += row * res;
        }

        //        Eigen::Matrix<double, 6, 1> x = JtJ.ldlt().solve(Jtb);
        Eigen::Matrix<double, 6, 1> x = JtJ.selfadjointView<Eigen::Upper>().ldlt().solve(Jtb);
        src                           = SE3::exp(x) * src;
    }
    return src;
}

inline Mat3 covR(const Mat3& R, double e)
{
    Mat3 cov;
    cov << 1, 0, 0, 0, 1, 0, 0, 0, e;
    return R.transpose() * cov * R;
}


SE3 planeToPlane(const AlignedVector<Correspondence>& corrs, const SE3& guess, double covE, int innerIterations)
{
    SAIGA_ASSERT(corrs.size() >= 6);
    SE3 T = guess;

    Eigen::Matrix<double, 6, 6> JtOmegaJ;
    Eigen::Matrix<double, 6, 1> JtOmegatb;



    // Covariance matrices for ref and src
    AlignedVector<Mat3> c0s, c1s;
    c0s.reserve(corrs.size());
    c1s.reserve(corrs.size());
    for (size_t i = 0; i < corrs.size(); ++i)
    {
        auto& corr = corrs[i];

        Mat3 R0 = onb(corr.refNormal).transpose();
        Mat3 R1 = onb(corr.srcNormal).transpose();

        Mat3 C0 = covR(R0, covE);
        Mat3 C1 = covR(R1, covE);

        c0s.push_back(C0);
        c1s.push_back(C1);
    }


    for (int k = 0; k < innerIterations; ++k)
    {
        JtOmegaJ.setZero();
        JtOmegatb.setZero();

        for (size_t i = 0; i < corrs.size(); ++i)
        {
            auto& corr = corrs[i];
            Eigen::Matrix<double, 3, 6> Jrow;

            Vec3 sp = T * corr.srcPoint;

            Jrow.block<3, 3>(0, 0) = Mat3::Identity();
            Jrow.block<3, 3>(0, 3) = -skew(sp);

            auto C0 = c0s[i];
            auto C1 = c1s[i];

            Mat3 Rt   = T.so3().matrix();
            Mat3 info = (C0 + Rt * C1 * Rt.transpose()).inverse();

            Vec3 res = corr.refPoint - sp;

            // use weight
            Jrow *= corr.weight;
            res *= corr.weight;

            JtOmegaJ += Jrow.transpose() * info * Jrow;
            JtOmegatb += Jrow.transpose() * info.transpose() * res;
        }
        Eigen::Matrix<double, 6, 1> x = JtOmegaJ.ldlt().solve(JtOmegatb);
        T                             = SE3::exp(x) * T;
    }
    return T;
}



}  // namespace ICP
}  // namespace Saiga
