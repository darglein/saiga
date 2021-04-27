/**
 * Copyright (c) 2021 Darius Rückert
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

Quat orientationFromMixedMatrixUQ(const Mat3& M)
{
    // Closed-form solution of absolute orientation using unit quaternions
    // https://pdfs.semanticscholar.org/3120/a0e44d325c477397afcf94ea7f285a29684a.pdf

    // Lower triangle
    Mat4 N;
    N(0, 0) = M(0, 0) + M(1, 1) + M(2, 2);
    N(1, 0) = M(1, 2) - M(2, 1);
    N(2, 0) = M(2, 0) - M(0, 2);
    N(3, 0) = M(0, 1) - M(1, 0);

    N(1, 1) = M(0, 0) - M(1, 1) - M(2, 2);
    N(2, 1) = M(0, 1) + M(1, 0);
    N(3, 1) = M(2, 0) + M(0, 2);

    N(2, 2) = -M(0, 0) + M(1, 1) - M(2, 2);
    N(3, 2) = M(1, 2) + M(2, 1);

    N(3, 3) = -M(0, 0) - M(1, 1) + M(2, 2);
    //    N       = N.selfadjointView<Eigen::Upper>();

    //    Eigen::EigenSolver<Mat4> eigenSolver(N, true);

    // Only the lower triangular part of the input matrix is referenced.
    Eigen::SelfAdjointEigenSolver<Mat4> eigenSolver(N);

    int largestEV = 0;
    for (auto i = 1; i < 4; ++i)
    {
        if (eigenSolver.eigenvalues()(i) > eigenSolver.eigenvalues()(largestEV))
        {
            largestEV = i;
        }
    }

    int largestEV2 = 0;
    eigenSolver.eigenvalues().maxCoeff(&largestEV2);
    SAIGA_ASSERT(largestEV == largestEV2);

    Vec4 E = eigenSolver.eigenvectors().col(largestEV);


    Quat R(E(0), E(1), E(2), E(3));
    R = R.conjugate();
    R.normalize();
    if (R.w() < 0) R.coeffs() *= -1;
    return R;
    //    return R;
}

inline Quat orientationFromMixedMatrixSVD(const Mat3& M)
{
    // polar decomp
    // M = USV^T
    // R = UV^T
    Eigen::JacobiSVD svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);

    Vec3 S = Vec3::Ones(3);
    if (svd.matrixU().determinant() * svd.matrixV().determinant() < 0) S(2) = -1;

    Mat3 R = svd.matrixU() * S.asDiagonal() * svd.matrixV().transpose();

    Quat q = Quat(R).normalized();
    return q;
}

SE3 pointToPointDirect(const AlignedVector<Correspondence>& corrs, double* scale)
{
    auto cpy = corrs;

    // Compute center
    Vec3 meanRef(0, 0, 0);
    Vec3 meanSrc(0, 0, 0);
    for (auto c : corrs)
    {
        meanRef += c.refPoint;
        meanSrc += c.srcPoint;
    }
    meanRef /= corrs.size();
    meanSrc /= corrs.size();

    // Translate src to target and computed squared distance sum
    double refSumSq = 0;
    double srcSumSq = 0;
    for (auto& c : cpy)
    {
        c.refPoint -= meanRef;
        c.srcPoint -= meanSrc;
        refSumSq += c.refPoint.squaredNorm();
        srcSumSq += c.srcPoint.squaredNorm();
    }

    double S = 1;

    if (scale)
    {
        S      = sqrt(refSumSq / srcSumSq);
        *scale = S;
    }

    Vec3 t = meanRef - meanSrc;
    Mat3 M;
    M.setZero();
    for (auto c : cpy)
    {
        M += (c.refPoint) * (c.srcPoint).transpose();
    }

    Quat R;

    if (0)
        R = orientationFromMixedMatrixUQ(M);
    else
        R = orientationFromMixedMatrixSVD(M);

    t = meanRef - S * (R * meanSrc);
    return SE3(Quat(R), t);
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

SE3 alignMinimal(const Mat3& src, const Mat3& dst)
{
    const int n = src.cols();  // number of measurements

    // required for demeaning ...
    const double one_over_n = 1.0 / n;

    // computation of mean
    const Vec3 src_mean = src.rowwise().sum() * one_over_n;
    const Vec3 dst_mean = dst.rowwise().sum() * one_over_n;



    // demeaning of src and dst points
    const Mat3 src_demean = src.colwise() - src_mean;
    const Mat3 dst_demean = dst.colwise() - dst_mean;


    const Mat3 sigma = one_over_n * dst_demean * src_demean.transpose();

    Quat q = ICP::orientationFromMixedMatrixSVD(sigma);
    Vec3 t = dst_mean - (q * src_mean);
    return SE3(q, t);
}


}  // namespace ICP
}  // namespace Saiga
