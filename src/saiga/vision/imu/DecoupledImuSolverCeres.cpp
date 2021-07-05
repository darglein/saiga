/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "DecoupledImuScene.h"

#ifdef SAIGA_USE_CERES

#    include "saiga/core/time/all.h"
#    include "saiga/vision/ceres/CeresHelper.h"
#    include "saiga/vision/ceres/local_parameterization_se3.h"
#    include "saiga/vision/imu/CeresPreintegration.h"

#    include "ceres/autodiff_cost_function.h"
#    include "ceres/evaluation_callback.h"
#    include "ceres/local_parameterization.h"
namespace Saiga::Imu
{
struct ImuErrorAD
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ImuErrorAD(const NavState& state1, const NavState& state2, const NavEdge& edge, const Vec3& weight_PVR)
        : state1(state1), state2(state2), edge(edge), weight_PVR(weight_PVR)
    {
    }

    using CostType         = ImuErrorAD;
    using CostFunctionType = ceres::AutoDiffCostFunction<CostType, 9, 4, 1, 3, 3, 3, 3>;
    template <typename... Types>
    static CostFunctionType* create(const Types&... args)
    {
        return new CostFunctionType(new CostType(args...));
    }

    template <typename T>
    bool operator()(const T* const _g, const T* const _scale, const T* const _bias_acc, const T* const _bias_gyro,
                    const T* const _v1, const T* const _v2, T* _residual) const
    {
        // saiga residual

        // 2. Map parameters
        T scale = *_scale;
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> bias_gyro(_bias_gyro);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> bias_acc(_bias_acc);
        Eigen::Map<const Eigen::Quaternion<T>> g_R(_g);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> Vi(_v1);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> Vj(_v2);
        Eigen::Map<Eigen::Matrix<T, 9, 1>> residual(_residual);

        Vec3 ref_g(0, 9.81, 0);
        Eigen::Matrix<T, 3, 1> g = g_R * ref_g.cast<T>();

        auto& sequence = edge.data;
        CeresPreintegration<T> preint(bias_gyro, bias_acc);
        preint.IntegrateMidPoint(*sequence);

        Sophus::SE3<T> p1 = state1.pose.cast<T>();
        Sophus::SE3<T> p2 = state2.pose.cast<T>();

        VelocityAndBiasBase<T> vb_i;
        vb_i.acc_bias  = bias_acc;
        vb_i.gyro_bias = bias_gyro;
        vb_i.velocity  = Vi;

        VelocityAndBiasBase<T> vb_j;
        vb_j.velocity = Vj;
        residual      = preint.Residual(vb_i, p1, vb_j, p2, g, scale);

        residual.template segment<3>(0) *= weight * T(weight_PVR(0));
        residual.template segment<3>(3) *= weight * T(weight_PVR(1));
        residual.template segment<3>(6) *= weight * T(weight_PVR(2));
        return true;



        return true;
    }

    const NavState& state1;
    const NavState& state2;
    const NavEdge& edge;
    Vec3 weight_PVR;
    SE3 camera_to_body;
    double weight = 1;
};



struct ImuError : public ceres::SizedCostFunction<9, 4, 1, 3, 3, 3, 3>
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ImuError(const NavState& state1, const NavState& state2, NavEdge& edge, const Vec3& weight_PVR)
        : state1(state1), state2(state2), edge(edge), weight_PVR(weight_PVR)
    {
    }

    // Params order:
    //  _bias_gyro, _bias_acc, _g,_v1,   _v2,_scale
    virtual bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
    {
        // saiga residual
        using T = double;

        Eigen::Map<const Eigen::Quaternion<T>> g_R(parameters[0]);
        T scale = parameters[1][0];
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> bias_acc(parameters[2]);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> bias_gyro(parameters[3]);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> Vi(parameters[4]);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> Vj(parameters[5]);


        Eigen::Map<Eigen::Matrix<T, 9, 1>> residual(residuals);


        Gravity g;
        g.R = SO3(g_R);



        //        Imu::Preintegration preint(bias_gyro, bias_acc);
        //        preint.IntegrateMidPoint(*edge.data);

        Sophus::SE3<T> p1 = state1.pose.cast<T>();
        Sophus::SE3<T> p2 = state2.pose.cast<T>();



        VelocityAndBias delta;


        if (jacobians == nullptr)
        {
            residual = edge.preint->ImuError(delta, Vi, p1, Vj, p2, g, scale, weight_PVR * edge.weight_pvr);

            return true;
        }



        Matrix<double, 9, 3> J_biasa, J_biasg;
        Matrix<double, 9, 3> J_v1, J_v2;
        Matrix<double, 9, 1> J_scale;
        Matrix<double, 9, 3> J_g;
        residual = edge.preint->ImuError(delta, Vi, p1, Vj, p2, g, scale, weight_PVR * edge.weight_pvr, &J_biasa,
                                         &J_biasg, &J_v1, &J_v2, nullptr, nullptr, &J_scale, &J_g);


        if (jacobians[0])
        {
            // g
            //            SAIGA_EXIT_ERROR("not implemented");
            Eigen::Map<Eigen::Matrix<T, 9, 4, Eigen::RowMajor>> m(jacobians[0]);
            m.setZero();
            m.block<9, 3>(0, 0) = J_g;
        }
        if (jacobians[1])
        {
            Eigen::Map<Eigen::Matrix<T, 9, 1>> m(jacobians[1]);
            m = J_scale;
        }
        if (jacobians[2])
        {
            // bias acc
            Eigen::Map<Eigen::Matrix<T, 9, 3, Eigen::RowMajor>> m(jacobians[2]);
            m = J_biasa;
        }
        if (jacobians[3])
        {
            // bias gyro
            Eigen::Map<Eigen::Matrix<T, 9, 3, Eigen::RowMajor>> m(jacobians[3]);
            m = J_biasg;
        }

        if (jacobians[4])
        {
            // vi
            Eigen::Map<Eigen::Matrix<T, 9, 3, Eigen::RowMajor>> m(jacobians[4]);
            m = J_v1;
        }
        if (jacobians[5])
        {
            Eigen::Map<Eigen::Matrix<T, 9, 3, Eigen::RowMajor>> m(jacobians[5]);
            m = J_v2;
        }



        return true;
    }

    const NavState& state1;
    const NavState& state2;
    const NavEdge& edge;
    Vec3 weight_PVR;
    SE3 camera_to_body;
    double weight = 1;
};

struct BiasChange
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    BiasChange(double weight_gyro, double weight_acc) : weight_gyro(weight_gyro), weight_acc(weight_acc) {}

    using CostType         = BiasChange;
    using CostFunctionType = ceres::AutoDiffCostFunction<CostType, 6, 3, 3, 3, 3>;
    template <typename... Types>
    static CostFunctionType* create(const Types&... args)
    {
        return new CostFunctionType(new CostType(args...));
    }

    template <typename T>
    bool operator()(const T* const _bg1, const T* const _ba1, const T* const _bg2, const T* const _ba2,
                    T* _residual) const
    {
        // 2. Map parameters
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> bg1(_bg1);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> ba1(_ba1);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> bg2(_bg2);
        Eigen::Map<const Eigen::Matrix<T, 3, 1>> ba2(_ba2);
        Eigen::Map<Eigen::Matrix<T, 6, 1>> residual(_residual);

        residual.template segment<3>(0) = (bg1 - bg2) * T(weight_gyro);
        residual.template segment<3>(3) = (ba1 - ba2) * T(weight_acc);

        return true;
    }

    double weight_gyro, weight_acc;
};

struct PreintCallBack : public ceres::EvaluationCallback
{
    virtual void PrepareForEvaluation(bool evaluate_jacobians, bool new_evaluation_point) override
    {
        // std::cout << "callback " << evaluate_jacobians << " " << new_evaluation_point << std::endl;
        if (new_evaluation_point)
        {
            //            SAIGA_BLOCK_TIMER();
            for (auto& e : scene->edges)
            {
                if (global_bias)
                {
                    *e.preint = Imu::Preintegration(scene->global_bias_gyro, scene->global_bias_acc);
                }
                else
                {
                    *e.preint = Imu::Preintegration(scene->states[e.from].velocity_and_bias);
                }
                e.preint->IntegrateMidPoint(*e.data, true);
            }
        }
    }
    DecoupledImuScene* scene;
    bool global_bias;
};


void DecoupledImuScene::SolveCeres(const SolverOptions& params, bool ad)
{
    // Setup solver options
    ceres::Problem::Options problemOptions;
    //    problemOptions.evaluation_callback
    problemOptions.local_parameterization_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;


    PreintCallBack cb;
    cb.scene       = this;
    cb.global_bias = params.use_global_bias;


    ceres::Problem problem(problemOptions);
    OptimizationOptions optimizationOptions;
    optimizationOptions.debug                  = false;
    optimizationOptions.debugOutput            = false;
    optimizationOptions.maxIterativeIterations = 2000;
    optimizationOptions.maxIterations          = params.max_its;
    optimizationOptions.solverType             = OptimizationOptions::SolverType::Direct;
    ceres::Solver::Options ceres_options       = make_options(optimizationOptions);

    if (!ad)
    {
        ceres_options.evaluation_callback = &cb;
    }
    //    ceres_options.callbacks


    ceres::QuaternionParameterization gravitiy_parameterization;
    Sophus::test::LocalParameterizationSO3Lie gravitiy_parameterization_lie;
    //    Quat g_r = gravity.R.unit_quaternion();


    {
        if (ad)
        {
            problem.AddParameterBlock(gravity.R.data(), 4, &gravitiy_parameterization);
        }
        else
        {
            problem.AddParameterBlock(gravity.R.data(), 4, &gravitiy_parameterization_lie);
        }

        // Global params
        problem.AddParameterBlock(&scale, 1, nullptr);

        if (params.use_global_bias)
        {
            problem.AddParameterBlock(global_bias_acc.data(), 3, nullptr);
            problem.AddParameterBlock(global_bias_gyro.data(), 3, nullptr);

            if (!(params.solver_flags & IMU_SOLVE_BA)) problem.SetParameterBlockConstant(global_bias_acc.data());
            if (!(params.solver_flags & IMU_SOLVE_BG)) problem.SetParameterBlockConstant(global_bias_gyro.data());
        }

        if (!(params.solver_flags & IMU_SOLVE_SCALE)) problem.SetParameterBlockConstant(&scale);
        if (!(params.solver_flags & IMU_SOLVE_GRAVITY)) problem.SetParameterBlockConstant(gravity.R.data());
    }



    for (auto& s : states)
    {
        auto ba = s.velocity_and_bias.acc_bias.data();
        auto bg = s.velocity_and_bias.gyro_bias.data();
        auto v  = s.velocity_and_bias.velocity.data();

        problem.AddParameterBlock(ba, 3, nullptr);
        problem.AddParameterBlock(bg, 3, nullptr);
        problem.AddParameterBlock(v, 3, nullptr);

        if (s.constant || !(params.solver_flags & IMU_SOLVE_BA)) problem.SetParameterBlockConstant(ba);
        if (s.constant || !(params.solver_flags & IMU_SOLVE_BG)) problem.SetParameterBlockConstant(bg);
        if (s.constant || !(params.solver_flags & IMU_SOLVE_VELOCITY)) problem.SetParameterBlockConstant(v);
    }



    for (int i = 0; i < edges.size(); ++i)
    {
        auto& e  = edges[i];
        auto& s1 = states[e.from];
        auto& s2 = states[e.to];

        double* v1 = s1.velocity_and_bias.velocity.data();
        double* v2 = s2.velocity_and_bias.velocity.data();

        double *vbg, *vba;
        if (params.use_global_bias)
        {
            vbg = global_bias_gyro.data();
            vba = global_bias_acc.data();
        }
        else
        {
            vbg = s1.velocity_and_bias.gyro_bias.data();
            vba = s1.velocity_and_bias.acc_bias.data();

            auto vbg2 = s2.velocity_and_bias.gyro_bias.data();
            auto vba2 = s2.velocity_and_bias.acc_bias.data();

            //            double wg = 1.0 / (Snake::imu.omega_random_walk * sqrt(Snake::imu.frequency) * sqrt(s2.time -
            //            s1.time)); double wa =
            //                1.0 / (Snake::imu.acceleration_random_walk * sqrt(Snake::imu.frequency) * sqrt(s2.time -
            //                s1.time));

            //            std::cout << "random walk " << wg << " / " << wa << std::endl;


            if ((params.solver_flags & IMU_SOLVE_BA) || (params.solver_flags & IMU_SOLVE_BG))
            {
                double dt          = std::abs(s2.time - s1.time);
                double weight_time = 1.0 / sqrt(dt);
                SAIGA_ASSERT(std::isfinite(weight_time));
                auto f_change = BiasChange::create(weight_time * weight_change_g * e.weight_bias(1),
                                                   weight_time * weight_change_a * e.weight_bias(0));
                problem.AddResidualBlock(f_change, nullptr, vbg, vba, vbg2, vba2);
            }
        }


        if (ad)
        {
            auto f = ImuErrorAD::create(s1, s2, e, WeightPVR());
            problem.AddResidualBlock(f, nullptr, gravity.R.data(), &scale, vba, vbg, v1, v2);
        }
        else
        {
            auto f = new ImuError(s1, s2, e, WeightPVR());
            problem.AddResidualBlock(f, nullptr, gravity.R.data(), &scale, vba, vbg, v1, v2);
        }



#    if 0
        CostGyroBias c(s1, s2, e, Snake::mono_intrinsics.camera_to_body);
        Matrix<double, 9, 1> residual;
        c(vbg, vba, g_r.coeffs().data(), v1, v2, &scale, residual.data());
        std::cout << i << " " << residual.transpose() << std::endl;
#    endif
    }



    //    double bef = chi2();

    //    printDebugJacobi(problem, 40, true);
    //    return;

    OptimizationResults result = ceres_solve(ceres_options, problem);



    //    std::cout << g_r << std::endl;
    // gravity.R = SO3(g_r);

    //    double aft = chi2();

#    if 0
    std::cout << "imu solve " << bef << " -> " << aft << std::endl;
    std::cout << "imu solve " << result.cost_initial << " -> " << result.cost_final << " in " << result.total_time
              << " ( " << result.linear_solver_time << ")" << std::endl;

    //    if (scale != 1)
    {
        if (!params.use_global_bias)
        {
            Vec3 mean_gyro = Vec3::Zero();
            Vec3 mean_acc  = Vec3::Zero();
            for (auto s : states)
            {
                mean_gyro += s.velocity_and_bias.gyro_bias;
                mean_acc += s.velocity_and_bias.acc_bias;
            }
            mean_gyro /= states.size();
            mean_acc /= states.size();

            global_bias_acc  = mean_acc;
            global_bias_gyro = mean_gyro;
        }

        std::cout << "Solve ";
        // std::cout << "Ceres IMU Solver. " << result.cost_initial << " -> " << result.cost_final << std::endl;
        std::cout << " BG: " << global_bias_gyro.transpose();
        std::cout << " BA: " << global_bias_acc.transpose();
        std::cout << " G: " << gravity.Get().transpose();
        std::cout << " S: " << scale << std::endl;
    }
#    endif

    //    for (auto s : states)
    //    {
    //        std::cout << s.velocity.transpose() << std::endl;
    //    }
}



}  // namespace Saiga::Imu

#endif
