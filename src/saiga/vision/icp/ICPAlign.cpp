/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "ICPAlign.h"


namespace Saiga {
namespace ICP {

SE3 pointToPoint(const std::vector<Correspondence> &corrs, const SE3 &guess)
{
    SE3 T = guess;
    Eigen::Matrix<double,6,6> JtJ;
    Eigen::Matrix<double,6,1> Jtb;
    JtJ.setZero();
    Jtb.setZero();

    for(size_t i = 0; i < corrs.size(); ++i)
    {
        auto& corr = corrs[i];

        Vec3 sp = T * corr.srcPoint;


        Eigen::Matrix<double,3,6> Jrow;
        Jrow.block<3,3>(0,0) = Mat3::Identity();
        Jrow.block<3,3>(0,3) = -skew(sp);


        Vec3 res = corr.refPoint - sp;

        // use weight
        Jrow *= corr.weight;
        res *= corr.weight;

        JtJ += Jrow.transpose() * Jrow;
        Jtb += Jrow.transpose() * res;
    }
    Eigen::Matrix<double,6,1> x = JtJ.ldlt().solve(Jtb);
    T = SE3::exp(x) * T;
    return T;
}

SE3 pointToPlane(const std::vector<Correspondence> &corrs, const SE3 &guess, int innerIterations)
{
    SE3 T = guess;
    Eigen::Matrix<double,6,6> JtJ;
    Eigen::Matrix<double,6,1> Jtb;


    for(int k = 0; k < innerIterations; ++k)
    {
        JtJ.setZero();
        Jtb.setZero();

        for(size_t i = 0; i < corrs.size(); ++i)
        {
            auto& corr = corrs[i];

            Vec3 sp = T * corr.srcPoint;

            Eigen::Matrix<double,6,1> row;
            row.head<3>() = corr.refNormal;
            row.tail<3>() = sp.cross(corr.refNormal);
            Vec3 di = corr.refPoint - sp;
            double res = corr.refNormal.dot(di);

            // use weight
            row *= corr.weight;
            res *= corr.weight;

            JtJ += row * row.transpose();
            Jtb += row * res;
        }
        Eigen::Matrix<double,6,1> x = JtJ.ldlt().solve(Jtb);
        T = SE3::exp(x) * T;
    }
    return T;
}

inline Mat3 covR(Mat3 R, double e)
{
    Mat3 cov;
    cov  << 1, 0, 0,
            0, 1, 0,
            0, 0, e;
    return R.transpose()*cov*R;
}


SE3 planeToPlane(const std::vector<Correspondence> &corrs, const SE3 &guess, double covE, int innerIterations)
{
    SE3 T = guess;

    Eigen::Matrix<double,6,6> JtOmegaJ;
    Eigen::Matrix<double,6,1> JtOmegatb;



    // Covariance matrices for ref and src
    std::vector<Mat3> c0s, c1s;
    c0s.reserve(corrs.size());
    c1s.reserve(corrs.size());
    for(size_t i = 0; i < corrs.size(); ++i)
    {
        auto& corr = corrs[i];

        Mat3 R0 = onb(corr.refNormal).transpose();
        Mat3 R1 = onb(corr.srcNormal).transpose();

        Mat3 C0 = covR(R0,covE);
        Mat3 C1 = covR(R1,covE);

        c0s.push_back(C0);
        c1s.push_back(C1);
    }


    for(int k = 0; k < innerIterations; ++k)
    {
        JtOmegaJ.setZero();
        JtOmegatb.setZero();

        for(size_t i = 0; i < corrs.size(); ++i)
        {
            auto& corr = corrs[i];
            Eigen::Matrix<double,3,6> Jrow;

            Vec3 sp = T * corr.srcPoint;

            Jrow.block<3,3>(0,0) = Mat3::Identity();
            Jrow.block<3,3>(0,3) = -skew(sp);

            auto C0 = c0s[i];
            auto C1 = c1s[i];

            Mat3 Rt = T.so3().matrix();
            Mat3 info = ( C0 + Rt * C1 * Rt.transpose() ).inverse();

            Vec3 res = corr.refPoint - sp;

            // use weight
            Jrow *= corr.weight;
            res *= corr.weight;

            JtOmegaJ  += Jrow.transpose() * info * Jrow;
            JtOmegatb += Jrow.transpose() * info.transpose() * res;
        }
        Eigen::Matrix<double,6,1> x = JtOmegaJ.ldlt().solve(JtOmegatb);
        T = SE3::exp(x) * T;
    }
    return T;
}



}
}
