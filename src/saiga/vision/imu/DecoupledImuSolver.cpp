/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "DecoupledImuSolver.h"

#include "saiga/core/time/all.h"
#include "saiga/core/util/Algorithm.h"
#include "saiga/vision/util/LM.h"

namespace Saiga::Imu
{
void DecoupledImuSolver::init()
{
    Eigen::setNbThreads(1);

    auto& scene = *_scene;
    N           = scene.states.size();

    SAIGA_ASSERT(!params.use_global_bias);

    num_params = N + 1;
    S.setZero();
    S.resize(num_params, num_params);

    b.resize(num_params);
    x.resize(num_params);

    // edges + diagonal + top row with diag
    non_zeros = scene.edges.size() + num_params + (num_params - 1);
    S.resizeNonZeros(non_zeros);
    //    S.reserve(non_zeros);

    for (int i = 0; i < S.outerSize(); ++i)
    {
        SAIGA_ASSERT(S.outerIndexPtr()[i] == 0);
    }

    // Compute outer structure pointer
    // Note: the row is smaller than the column, which means we are in the upper right part of the matrix.
    for (auto& e : scene.edges)
    {
        int i = e.from + 1;
        int j = e.to + 1;
        SAIGA_ASSERT(i != j);
        SAIGA_ASSERT(i < j);
        S.outerIndexPtr()[i]++;
    }

    // make room for the diagonal element
    for (int i = 0; i < S.rows(); ++i)
    {
        S.outerIndexPtr()[i]++;
    }

    // first row
    S.outerIndexPtr()[0] += num_params - 1;

    int n = exclusive_scan(S.outerIndexPtr(), S.outerIndexPtr() + S.outerSize(), S.outerIndexPtr(), 0);
    S.outerIndexPtr()[S.outerSize()] = non_zeros;


    SAIGA_ASSERT(n == non_zeros);


    // insert diagonal index
    for (int i = 0; i < S.rows(); ++i)
    {
        int offseti                = S.outerIndexPtr()[i];
        S.innerIndexPtr()[offseti] = i;
    }

    // insert top row
    for (int i = 0; i < S.cols(); ++i)
    {
        //        int offseti                = S.outerIndexPtr()[i];
        S.innerIndexPtr()[0 + i] = i;
    }

    // Precompute the offset in the sparse matrix for every edge
    edgeOffsets.clear();
    edgeOffsets.reserve(scene.edges.size());
    std::vector<int> localOffsets(S.rows(), 1);
    for (auto& e : scene.edges)
    {
        int i = e.from + 1;
        int j = e.to + 1;

        int li = localOffsets[i]++;

        int offseti = S.outerIndexPtr()[i] + li;


        SAIGA_ASSERT(offseti < S.nonZeros());
        SAIGA_ASSERT(j < S.rows());
        S.innerIndexPtr()[offseti] = j;


        edgeOffsets.emplace_back(offseti);
    }

    // ====
    states_without_preint.clear();
    std::vector<bool> has_preint(scene.states.size(), false);
    for (auto& e : scene.edges)
    {
        has_preint[e.from] = true;
    }
    for (int i = 0; i < scene.states.size(); ++i)
    {
        if (!has_preint[i]) states_without_preint.push_back(i);
    }

    solver.Init();
}

double DecoupledImuSolver::computeQuadraticForm()
{
    auto& scene = *_scene;

    b.setZero();

    for (int i = 0; i < non_zeros; ++i)
    {
        S.valuePtr()[i].get().setZero();
    }

    Matrix<double, 9, 3> _J_biasa, _J_biasg;
    Matrix<double, 9, 3> _J_v1, _J_v2;
    Matrix<double, 9, 1> _J_scale;
    Matrix<double, 9, 3> _J_g;



    Matrix<double, 9, 1>* J_scale = (params.solver_flags & IMU_SOLVE_SCALE) ? &_J_scale : nullptr;
    Matrix<double, 9, 3>* J_g     = (params.solver_flags & IMU_SOLVE_GRAVITY) ? &_J_g : nullptr;


    Matrix<double, 6, 6> J_a_g_i, J_a_g_j;

    double chi2 = 0;
    for (int edge_id = 0; edge_id < scene.edges.size(); ++edge_id)
    {
        auto& e = scene.edges[edge_id];

        int i = e.from;
        int j = e.to;

        auto& s1 = scene.states[i];
        auto& s2 = scene.states[j];


        Matrix<double, 9, 3>* J_biasa = (!s1.constant && (params.solver_flags & IMU_SOLVE_BA)) ? &_J_biasa : nullptr;
        Matrix<double, 9, 3>* J_biasg = (!s1.constant && (params.solver_flags & IMU_SOLVE_BG)) ? &_J_biasg : nullptr;
        Matrix<double, 9, 3>* J_v1    = (!s1.constant && (params.solver_flags & IMU_SOLVE_VELOCITY)) ? &_J_v1 : nullptr;
        Matrix<double, 9, 3>* J_v2    = (!s2.constant && (params.solver_flags & IMU_SOLVE_VELOCITY)) ? &_J_v2 : nullptr;


        auto& Vi = s1.velocity_and_bias.velocity;
        auto& Vj = s2.velocity_and_bias.velocity;
        auto& p1 = s1.pose;
        auto& p2 = s2.pose;


        Vec9 res = e.preint->ImuError(s1.delta_bias, Vi, p1, Vj, p2, scene.gravity, scene.scale,
                                      scene.WeightPVR() * e.weight_pvr, J_biasa, J_biasg, J_v1, J_v2, nullptr, nullptr,
                                      J_scale, J_g);


        Matrix<double, 9, 9> J1, J2;
        J1.setZero();
        J2.setZero();



        if (J_biasa) J1.block<9, 3>(0, 0) = *J_biasa;
        if (J_biasg) J1.block<9, 3>(0, 3) = *J_biasg;

        if (J_v1) J1.block<9, 3>(0, 6) = *J_v1;
        if (J_v2) J2.block<9, 3>(0, 6) = *J_v2;

        int offset_i = i + 1;
        int offset_j = j + 1;

        SAIGA_ASSERT(offset_i < b.rows());
        SAIGA_ASSERT(offset_j < b.rows());

        SAIGA_ASSERT(offset_i >= 0 && offset_i < S.outerSize());
        SAIGA_ASSERT(offset_j >= 0 && offset_j < S.outerSize());
        SAIGA_ASSERT(edgeOffsets[edge_id] >= 0 && edgeOffsets[edge_id] < S.nonZeros());

        auto& target_ii = S.valuePtr()[S.outerIndexPtr()[offset_i]].get();
        auto& target_jj = S.valuePtr()[S.outerIndexPtr()[offset_j]].get();
        auto& target_ij = S.valuePtr()[edgeOffsets[edge_id]].get();
        auto& target_ir = b(offset_i).get();
        auto& target_jr = b(offset_j).get();


        target_ii += J1.transpose() * J1;
        target_jj += J2.transpose() * J2;
        target_ij += J1.transpose() * J2;

        target_ir -= J1.transpose() * res;
        target_jr -= J2.transpose() * res;


        if (J_g)
        {
            auto target_gg = S.valuePtr()[0].get().block<3, 3>(0, 0);
            target_gg += (*J_g).transpose() * (*J_g);

            auto target_gr = b(0).get().segment<3>(0, 0);
            target_gr -= (*J_g).transpose() * (res);
            if (J_scale)
            {
                auto target_gs = S.valuePtr()[0].get().block<3, 1>(0, 3);
                target_gs += (*J_g).transpose() * (*J_scale);
            }

            if (J_biasa)
            {
                auto target_gba = S.valuePtr()[offset_i].get().block<3, 3>(0, 0);
                target_gba += (*J_g).transpose() * (*J_biasa);
            }
            if (J_biasg)
            {
                auto target_gbg = S.valuePtr()[offset_i].get().block<3, 3>(0, 3);
                target_gbg += (*J_g).transpose() * (*J_biasg);
            }

            if (J_v1)
            {
                auto target_gv1 = S.valuePtr()[offset_i].get().block<3, 3>(0, 6);
                target_gv1 += (*J_g).transpose() * (*J_v1);
            }

            if (J_v2)
            {
                auto target_gv2 = S.valuePtr()[offset_j].get().block<3, 3>(0, 6);
                target_gv2 += (*J_g).transpose() * (*J_v2);
            }
        }
        if (J_scale)
        {
            auto target_ss = S.valuePtr()[0].get().block<1, 1>(3, 3);
            target_ss += (*J_scale).transpose() * (*J_scale);

            auto target_sr = b(0).get().segment<1>(3);
            target_sr -= (*J_scale).transpose() * (res);

            if (J_biasa)
            {
                auto target_sba = S.valuePtr()[offset_i].get().block<1, 3>(3, 0);
                target_sba += (*J_scale).transpose() * (*J_biasa);
            }
            if (J_biasg)
            {
                auto target_sbg = S.valuePtr()[offset_i].get().block<1, 3>(3, 3);
                target_sbg += (*J_scale).transpose() * (*J_biasg);
            }

            if (J_v1)
            {
                auto target_sv1 = S.valuePtr()[offset_i].get().block<1, 3>(3, 6);
                target_sv1 += (*J_scale).transpose() * (*J_v1);
            }

            if (J_v2)
            {
                auto target_sv2 = S.valuePtr()[offset_j].get().block<1, 3>(3, 6);
                target_sv2 += (*J_scale).transpose() * (*J_v2);
            }
        }

        double r = res.squaredNorm();
        SAIGA_ASSERT(std::isfinite(r));

        if ((params.solver_flags & IMU_SOLVE_BA) || (params.solver_flags & IMU_SOLVE_BG))
        {
            Vec6 res_bias_change = e.preint->BiasChangeError(
                s1.velocity_and_bias, s1.delta_bias, s2.velocity_and_bias, s2.delta_bias,
                scene.weight_change_a * e.weight_bias(0), scene.weight_change_g * e.weight_bias(1), &J_a_g_i, &J_a_g_j);


            if (!s1.constant)
            {
                target_ii.block<6, 6>(0, 0) += J_a_g_i.transpose() * J_a_g_i;
                target_ir.segment<6>(0) -= J_a_g_i.transpose() * res_bias_change;
            }
            if (!s2.constant)
            {
                target_jj.block<6, 6>(0, 0) += J_a_g_j.transpose() * J_a_g_j;
                target_jr.segment<6>(0) -= J_a_g_j.transpose() * res_bias_change;
            }
            if (!s1.constant && !s2.constant)
            {
                target_ij.block<6, 6>(0, 0) += J_a_g_i.transpose() * J_a_g_j;
            }


            r += res_bias_change.squaredNorm();
            SAIGA_ASSERT(std::isfinite(r));
        }



        chi2 += r;
    }


#if 0
    auto JtJ = expand(S);
    auto Jtb = expand(b);
    std::cout << "[Gradient Jtb]" << std::endl;
    std::cout << Jtb.transpose() << std::endl;
    std::cout << "|g| = " << Jtb.norm() << std::endl << std::endl;

    std::cout << "[JtJ]" << std::endl;

    JtJ = JtJ.selfadjointView<Eigen::Upper>();
    std::cout << JtJ << std::endl << std::endl;
    std::cout << "|J| = " << JtJ.norm() << std::endl;
#endif
    return chi2;
}

void DecoupledImuSolver::addLambda(double lambda)
{
#if 0
    double min_lm_diagonal = 1e-6;
    double max_lm_diagonal = 1e32;

    for (int k = 0; k < JtJ.rows(); ++k)
    {
        auto& value = JtJ.diagonal()(k);
        value       = value + lambda * value;
        value       = clamp(value, min_lm_diagonal, max_lm_diagonal);
    }
#else
    // apply lm
    for (int i = 0; i < S.rows(); ++i)
    {
        auto& d = S.valuePtr()[S.outerIndexPtr()[i]].get();
        Saiga::applyLMDiagonalInner(d, lambda);
    }
#endif
}

void DecoupledImuSolver::RecomputePreint(bool always)
{
    //    SAIGA_BLOCK_TIMER();
    int rec     = 0;
    auto& scene = *_scene;
    for (int i = 0; i < scene.edges.size(); ++i)
    {
        auto& e = scene.edges[i];
        auto& s = scene.states[e.from];

        if (always || s.delta_bias.acc_bias.squaredNorm() > params.bias_recompute_delta_squared ||
            s.delta_bias.gyro_bias.squaredNorm() > params.bias_recompute_delta_squared)
        {
            s.velocity_and_bias.acc_bias += s.delta_bias.acc_bias;
            s.velocity_and_bias.gyro_bias += s.delta_bias.gyro_bias;
            s.delta_bias = VelocityAndBias();

            *e.preint = Imu::Preintegration(s.velocity_and_bias);
            e.preint->IntegrateMidPoint(*e.data, true);
            rec++;
        }
    }


    for (auto is : states_without_preint)
    {
        auto& s = scene.states[is];

        s.velocity_and_bias.acc_bias += s.delta_bias.acc_bias;
        s.velocity_and_bias.gyro_bias += s.delta_bias.gyro_bias;
        s.delta_bias = VelocityAndBias();
    }
    //    std::cout << "Recomputed " << rec << " / " << scene.edges.size() << std::endl;
}


bool DecoupledImuSolver::addDelta()
{
    auto& scene = *_scene;



    Vec3 delta_g   = x(0).get().segment<3>(0);
    double delta_s = x(0).get()(3);

    scene.scale += delta_s;
    scene.gravity.R = Sophus::SO3d::exp(delta_g) * scene.gravity.R;

    for (int i = 0; i < scene.states.size(); ++i)
    {
        auto& s = scene.states[i];
        if (s.constant) continue;

        int offset_i  = i + 1;
        auto delta_ba = x(offset_i).get().segment<3>(0);
        auto delta_bg = x(offset_i).get().segment<3>(0 + 3);
        auto delta_v  = x(offset_i).get().segment<3>(0 + 6);

        if (params.solver_flags & IMU_SOLVE_VELOCITY) s.velocity_and_bias.velocity += delta_v;
        if (params.solver_flags & IMU_SOLVE_BA) s.delta_bias.acc_bias += delta_ba;
        if (params.solver_flags & IMU_SOLVE_BG) s.delta_bias.gyro_bias += delta_bg;
    }

    RecomputePreint(false);
    return true;
}

void DecoupledImuSolver::revertDelta()
{
    auto& scene = *_scene;

    Vec3 delta_g   = x(0).get().segment<3>(0);
    double delta_s = x(0).get()(3);
    scene.scale -= delta_s;
    scene.gravity.R = Sophus::SO3d::exp(delta_g).inverse() * scene.gravity.R;


    for (int i = 0; i < scene.states.size(); ++i)
    {
        auto& s = scene.states[i];
        if (s.constant) continue;

        int offset_i  = i + 1;
        auto delta_ba = x(offset_i).get().segment<3>(0);
        auto delta_bg = x(offset_i).get().segment<3>(0 + 3);
        auto delta_v  = x(offset_i).get().segment<3>(0 + 6);


        if (params.solver_flags & IMU_SOLVE_VELOCITY) s.velocity_and_bias.velocity -= delta_v;
        if (params.solver_flags & IMU_SOLVE_BA) s.delta_bias.acc_bias -= delta_ba;
        if (params.solver_flags & IMU_SOLVE_BG) s.delta_bias.gyro_bias -= delta_bg;
    }
    RecomputePreint(false);
}

void DecoupledImuSolver::solveLinearSystem()
{
    using namespace Eigen::Recursive;
    LinearSolverOptions loptions;

    loptions.maxIterativeIterations = optimizationOptions.maxIterativeIterations;
    loptions.iterativeTolerance     = optimizationOptions.iterativeTolerance;

    loptions.solverType = (optimizationOptions.solverType == OptimizationOptions::SolverType::Direct)
                              ? LinearSolverOptions::SolverType::Direct
                              : LinearSolverOptions::SolverType::Iterative;
    loptions.cholmod = true;


    solver.solve(S, x, b, loptions);
}

double DecoupledImuSolver::computeCost()
{
    auto& scene = *_scene;

    double chi2 = 0;
    for (int edge_id = 0; edge_id < scene.edges.size(); ++edge_id)
    {
        auto& e = scene.edges[edge_id];

        int i = e.from;
        int j = e.to;

        auto& s1 = scene.states[i];
        auto& s2 = scene.states[j];

        auto& Vi = s1.velocity_and_bias.velocity;
        auto& Vj = s2.velocity_and_bias.velocity;
        auto& p1 = s1.pose;
        auto& p2 = s2.pose;


        Vec9 res = e.preint->ImuError(s1.delta_bias, Vi, p1, Vj, p2, scene.gravity, scene.scale,
                                      scene.WeightPVR() * e.weight_pvr);

        double r = res.squaredNorm();

        if ((params.solver_flags & IMU_SOLVE_BA) || (params.solver_flags & IMU_SOLVE_BG))
        {
            Vec6 res_bias_change = e.preint->BiasChangeError(s1.velocity_and_bias, s1.delta_bias, s2.velocity_and_bias,
                                                             s2.delta_bias, scene.weight_change_a * e.weight_bias(0),
                                                             scene.weight_change_g * e.weight_bias(1));


            r += res_bias_change.squaredNorm();
        }
        SAIGA_ASSERT(std::isfinite(r));
        chi2 += r;
    }
    return chi2;
}

void DecoupledImuSolver::finalize()
{
    if (params.final_recompute)
    {
        RecomputePreint(true);
    }
}


}  // namespace Saiga::Imu
