/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "MultiViewICPAlign.h"

#include "saiga/core/time/timer.h"
#include "saiga/core/util/assert.h"


namespace Saiga
{
namespace ICP
{
void multiViewICPAlign(size_t N, const std::vector<std::pair<size_t, size_t>>& pairs,
                       const std::vector<AlignedVector<Correspondence>>& corrs, AlignedVector<SE3>& guesses,
                       int iterations)
{
#ifdef WITH_CERES
    Sophus::test::LocalParameterizationSE3* camera_parameterization = new Sophus::test::LocalParameterizationSE3;
    ceres::Problem problem;


    for (size_t i = 0; i < corrs.size(); ++i)
    {
        auto& p  = pairs[i];
        auto& cs = corrs[i];

        auto& se0 = guesses[p.first];
        auto& se1 = guesses[p.second];

        for (auto c : cs)
        {
            auto cost_function = CostICPPlane::create(c.refPoint, c.refNormal, c.srcPoint, c.weight);
            problem.AddResidualBlock(cost_function, nullptr, se0.data(), se1.data());
        }
    }

    for (auto& g : guesses)
    {
        problem.SetParameterization(g.data(), camera_parameterization);
    }



    ceres::Solver::Options options;
    //        options.min_trust_region_radius = 1;
    //        options.max_trust_region_radius = 10e51;
    //        options.initial_trust_region_radius = 10e50;
    //        options.min_trust_region_radius = 10e50;
    //        options.min_lm_diagonal = 10e-50;
    //        options.max_lm_diagonal = 10e-49;



    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations           = 5;
    ceres::Solver::Summary summaryTest;
    ceres::Solve(options, &problem, &summaryTest);
#else

    for (auto inner = 0; inner < iterations; ++inner)
    {
        Eigen::MatrixXd JtJ(N * 6, N * 6);
        Eigen::MatrixXd Jtb(N * 6, 1);
        JtJ.setZero();
        Jtb.setZero();
        //        SAIGA_BLOCK_TIMER;

        size_t asdf = 0;

        for (size_t i = 0; i < corrs.size(); ++i)
        {
            auto& p  = pairs[i];
            auto& cs = corrs[i];

            auto& ref = guesses[p.first];
            auto& src = guesses[p.second];


            for (size_t c = 0; c < cs.size(); ++c)
            {
                auto& corr = cs[c];



                Vec3 rp = ref * corr.refPoint;
                Vec3 rn = ref.so3() * corr.refNormal;
                Vec3 sp = src * corr.srcPoint;

                Vec3 di    = rp - sp;
                double res = rn.dot(di);


                Mat3 skewP = skew(rp);
                Mat3 skewN = skew(rn);

                // Derivative towards se0
                Eigen::Matrix<double, 6, 1> rowRef;
                rowRef.setZero();

                rowRef.head<3>() = -rn;
                rowRef.tail<3>() = skewP.transpose() * rn + skewN.transpose() * rp - skewN.transpose() * sp;

                // Derivative towards se1
                Eigen::Matrix<double, 6, 1> rowSrc;
                rowSrc.head<3>() = rn;
                rowSrc.tail<3>() = sp.cross(rn);

                // Use weight
                res *= corr.weight;
                rowRef *= corr.weight;
                rowSrc *= corr.weight;

                // fill jtj directly
                //                    Eigen::MatrixXd JfullRow(N * 6, 1);
                //                    JfullRow.setZero();
                //                    JfullRow.block(p.first * 6, 0, 6, 1) = rowRef;
                //                    JfullRow.block(p.second * 6, 0, 6, 1) = rowSrc;


                // Equivalent to: JtJ += JfullRow * JfullRow.transpose();
                //                    JtJ.block(p.first * 6, p.first * 6, 6, 6) += rowRef * rowRef.transpose();
                //                    JtJ.block(p.second * 6, p.second * 6, 6, 6) += rowSrc * rowSrc.transpose();
                //                    JtJ.block(p.first * 6, p.second * 6, 6, 6) += rowRef * rowSrc.transpose();
                //                    JtJ.block(p.second * 6, p.first * 6, 6, 6) += rowSrc * rowRef.transpose();

                //                    JtJ.block(p.first * 6, p.first * 6, 6, 6) += rowRef * rowRef.transpose();

                JtJ.block(p.first * 6, p.first * 6, 6, 6) +=
                    (rowRef * rowRef.transpose()).triangularView<Eigen::Upper>();

                JtJ.block(p.second * 6, p.second * 6, 6, 6) +=
                    (rowSrc * rowSrc.transpose()).triangularView<Eigen::Upper>();
                //                    JtJ.block(p.first * 6, p.second * 6, 6, 6) += rowRef * rowSrc.transpose();
                JtJ.block(p.first * 6, p.second * 6, 6, 6) += rowRef * rowSrc.transpose();
                //                    JtJ.block(p.second * 6, p.first * 6, 6, 6) += rowSrc * rowRef.transpose();

                //  Equivalent to: Jtb += JfullRow * res;
                Jtb.block(p.first * 6, 0, 6, 1) += rowRef * res;
                Jtb.block(p.second * 6, 0, 6, 1) += rowSrc * res;

                asdf++;
            }
        }

        //            Eigen::MatrixXd x = JtJ.ldlt().solve(Jtb);
        Eigen::MatrixXd x = JtJ.selfadjointView<Eigen::Upper>().ldlt().solve(Jtb);


        //        std::cout << J.block(0, 0, 20, 6) << std::endl;
        //        std::cout << x.transpose() << std::endl;

        for (size_t n = 0; n < N; ++n)
        {
            //                        auto t = x.segment(n * 6, 6);
            auto t = x.block(n * 6, 0, 6, 1);
            //            auto t = x;
            //            size_t n = 1;

            guesses[n] = SE3::exp(t) * guesses[n];
        }
    }


#endif
}


}  // namespace ICP
}  // namespace Saiga
