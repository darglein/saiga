/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "RecursiveArap.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/core/time/timer.h"
#include "saiga/core/util/Algorithm.h"
#include "saiga/vision/kernels/Robust.h"
#include "saiga/vision/util/HistogramImage.h"
#include "saiga/vision/util/LM.h"



namespace Saiga
{
void RecursiveArap::init()
{
    auto& scene = *arap;


    n = scene.vertices.size();
    b.resize(n);
    delta_x.resize(n);

    x_u.resize(n);
    oldx_u.resize(n);

    // Make a copy of the initial parameters
    int i = 0;
    for (auto& e : scene.vertices)
    {
        x_u[i++] = e;
    }

    // Compute structure of S
    S.resize(n, n);
    S.setZero();
    int N = scene.constraints.size() + n;
    S.reserve(N);

#if 1
    // Compute outer structure pointer
    for (auto& e : scene.constraints)
    {
        int i = e.ids.first;
        int j = e.ids.second;
        SAIGA_ASSERT(i != j);
        SAIGA_ASSERT(i < j);

        S.outerIndexPtr()[i]++;
    }

    // make room for the diagonal element
    for (int i = 0; i < n; ++i)
    {
        S.outerIndexPtr()[i]++;
    }

    exclusive_scan(S.outerIndexPtr(), S.outerIndexPtr() + S.outerSize(), S.outerIndexPtr(), 0);
    S.outerIndexPtr()[S.outerSize()] = N;


    // insert diagonal index
    for (int i = 0; i < n; ++i)
    {
        int offseti                = S.outerIndexPtr()[i];
        S.innerIndexPtr()[offseti] = i;
    }

    // Precompute the offset in the sparse matrix for every edge
    edgeOffsets.reserve(scene.constraints.size());
    std::vector<int> localOffsets(n, 1);
    for (auto& e : scene.constraints)
    {
        int i = e.ids.first;
        int j = e.ids.second;

        int li = localOffsets[i]++;

        int offseti = S.outerIndexPtr()[i] + li;

        S.innerIndexPtr()[offseti] = j;


        edgeOffsets.emplace_back(offseti);
    }

    // Create a sparsity histogram
    if (false && optimizationOptions.debugOutput)
    {
        HistogramImage img(n, n, 1024 * 8, 1024 * 8);
        for (auto& e : scene.constraints)
        {
            int i = e.ids.first;
            int j = e.ids.second;
            img.add(i, j, 1);
            img.add(j, i, 1);
            img.add(i, i, 1);
            img.add(j, j, 1);
        }
        img.writeBinary("arap.png");
    }
#endif
}

double RecursiveArap::computeQuadraticForm()
{
    auto& scene = *arap;


    b.setZero();

    // set diagonal elements of S to zero
    for (int i = 0; i < n; ++i)
    {
        S.valuePtr()[S.outerIndexPtr()[i]].get().setZero();
    }


    double chi2 = 0;

    // Add targets
    for (int k = 0; k < (int)scene.target_indices.size(); ++k)
    {
        int i = scene.target_indices[k];


        auto& target_ii = S.valuePtr()[S.outerIndexPtr()[i]].get();
        auto& target_ir = b(i).get();

        auto p = x_u[i];
        auto t = scene.target_positions[k];


        Vec3 res = p.translation() - t;

        Eigen::Matrix<T, 3, 6, Eigen::RowMajor> Jrowi;
        Jrowi.block<3, 3>(0, 0) = Mat3::Identity();
        Jrowi.block<3, 3>(0, 3) = Mat3::Zero();

        // JtJ
        target_ii += Jrowi.transpose() * Jrowi;

        // Jtb
        target_ir -= Jrowi.transpose() * res;

        auto c = res.squaredNorm();
        chi2 += c;
    }


    for (size_t k = 0; k < scene.constraints.size(); ++k)
    {
        auto& e       = scene.constraints[k];
        auto& offsets = edgeOffsets[k];
        int i         = e.ids.first;
        int j         = e.ids.second;

        double w_Reg = sqrt(e.weight);

        auto& target_ij = S.valuePtr()[offsets].get();
        auto& target_ii = S.valuePtr()[S.outerIndexPtr()[i]].get();
        auto& target_jj = S.valuePtr()[S.outerIndexPtr()[j]].get();
        auto& target_ir = b(i).get();
        auto& target_jr = b(j).get();


        auto pHat = x_u[i];
        auto qHat = x_u[j];
        {
            Vec3 R_eij = pHat.so3() * e.e_ij;
            Vec3 res   = w_Reg * (pHat.translation() - qHat.translation() - R_eij);

            Eigen::Matrix<T, 3, 6, Eigen::RowMajor> Jrowi;
            Jrowi.block<3, 3>(0, 0) = Mat3::Identity();
            Jrowi.block<3, 3>(0, 3) = skew(R_eij);
            Jrowi *= w_Reg;

            Eigen::Matrix<T, 3, 6, Eigen::RowMajor> Jrowj;
            Jrowj.block<3, 3>(0, 0) = -Mat3::Identity();
            Jrowj.block<3, 3>(0, 3) = Mat3::Zero();
            Jrowj *= w_Reg;

            // JtJ
            target_ij = Jrowi.transpose() * Jrowj;
            target_ii += Jrowi.transpose() * Jrowi;
            target_jj += Jrowj.transpose() * Jrowj;

            // Jtb
            target_ir -= Jrowi.transpose() * res;
            target_jr -= Jrowj.transpose() * res;

            auto c = res.squaredNorm();
            chi2 += c;
        }
        if (1)
        {
            Vec3 R_eji = qHat.so3() * (-e.e_ij);
            Vec3 res   = w_Reg * (qHat.translation() - pHat.translation() - R_eji);

            Eigen::Matrix<T, 3, 6, Eigen::RowMajor> Jrowi;
            Jrowi.block<3, 3>(0, 0) = -Mat3::Identity();
            Jrowi.block<3, 3>(0, 3) = Mat3::Zero();
            Jrowi *= w_Reg;

            Eigen::Matrix<T, 3, 6, Eigen::RowMajor> Jrowj;
            Jrowj.block<3, 3>(0, 0) = Mat3::Identity();
            Jrowj.block<3, 3>(0, 3) = skew(R_eji);
            Jrowj *= w_Reg;


            target_ij += (Jrowi.transpose() * Jrowj);
            target_ii += Jrowi.transpose() * Jrowi;
            target_jj += Jrowj.transpose() * Jrowj;

            // Jtb
            target_ir -= Jrowi.transpose() * res;
            target_jr -= Jrowj.transpose() * res;

            auto c = res.squaredNorm();
            chi2 += c;
        }
    }

    return chi2;
}

void RecursiveArap::addLambda(double lambda)
{
    // apply lm
    for (int i = 0; i < n; ++i)
    {
        auto& d = S.valuePtr()[S.outerIndexPtr()[i]].get();
        applyLMDiagonalInner(d, lambda);
    }
}

void RecursiveArap::revertDelta()
{
    x_u = oldx_u;
}

bool RecursiveArap::addDelta()
{
    oldx_u = x_u;

    for (int i = 0; i < n; ++i)
    {
        auto t = delta_x(i).get();
        x_u[i] = x_u[i] * SE3::exp(t);
    }
    return true;
}

void RecursiveArap::solveLinearSystem()
{
    using namespace Eigen::Recursive;
    LinearSolverOptions loptions;

    loptions.maxIterativeIterations = optimizationOptions.maxIterativeIterations;
    loptions.iterativeTolerance     = optimizationOptions.iterativeTolerance;

    loptions.solverType = (optimizationOptions.solverType == OptimizationOptions::SolverType::Direct)
                              ? LinearSolverOptions::SolverType::Direct
                              : LinearSolverOptions::SolverType::Iterative;



    solver.solve(S, delta_x, b, loptions);
}

double RecursiveArap::computeCost()
{
    auto& scene = *arap;

    double chi2 = 0;

    for (int k = 0; k < (int)scene.target_indices.size(); ++k)
    {
        int i    = scene.target_indices[k];
        auto p   = x_u[i];
        auto t   = scene.target_positions[k];
        Vec3 res = p.translation() - t;
        auto c   = res.squaredNorm();
        chi2 += c;
    }

    for (size_t k = 0; k < scene.constraints.size(); ++k)
    {
        auto& e = scene.constraints[k];
        int i   = e.ids.first;
        int j   = e.ids.second;


        auto pHat = x_u[i];
        auto qHat = x_u[j];
        {
            Vec3 R_eij = pHat.so3() * e.e_ij;
            Vec3 res   = sqrt(e.weight) * (pHat.translation() - qHat.translation() - R_eij);
            auto c     = res.squaredNorm();
            chi2 += c;
        }
        {
            Vec3 R_eji = qHat.so3() * (-e.e_ij);
            Vec3 res   = sqrt(e.weight) * (qHat.translation() - pHat.translation() - R_eji);
            auto c     = res.squaredNorm();
            chi2 += c;
        }
    }
    return chi2;
}

void RecursiveArap::finalize()
{
    auto& scene = *arap;

    int i = 0;
    for (auto& e : scene.vertices)
    {
        e = x_u[i++];
    }
}

}  // namespace Saiga
