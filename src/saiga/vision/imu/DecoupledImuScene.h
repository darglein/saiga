/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/imu/Preintegration.h"


namespace Saiga::Imu
{
struct SAIGA_VISION_API NavState
{
    SE3 pose;
    Imu::VelocityAndBias velocity_and_bias;

    Imu::VelocityAndBias delta_bias;
    double time;
    bool constant = false;
};

struct SAIGA_VISION_API NavEdge
{
    Preintegration* preint = nullptr;
    ImuSequence* data      = nullptr;
    int from, to;

    double weight_pvr = 1;

    // (acc,gyro)
    Vec2 weight_bias = Vec2(1, 1);
};

enum ImuSolverFlags
{
    IMU_SOLVE_BA       = 1,
    IMU_SOLVE_BG       = 2,
    IMU_SOLVE_VELOCITY = 4,
    IMU_SOLVE_GRAVITY  = 8,
    IMU_SOLVE_SCALE    = 16,
};

class SAIGA_VISION_API DecoupledImuScene
{
   public:
    struct SolverOptions
    {  // Solver options
        //        bool solve_bias_gyro = true;
        //        bool solve_bias_acc  = true;
        //        bool solve_velocity  = true;
        //        bool solve_gravity   = true;
        //        bool solve_scale     = false;
        int solver_flags = IMU_SOLVE_BA | IMU_SOLVE_BG | IMU_SOLVE_VELOCITY | IMU_SOLVE_GRAVITY;

        bool use_global_bias = false;
        int max_its          = 3;

        double bias_recompute_delta_squared = 0.01;

        bool final_recompute = true;
    };

    Vec3 global_bias_gyro = Vec3::Zero();
    Vec3 global_bias_acc  = Vec3::Zero();
    Gravity gravity;
    double scale = 1.0;

    std::vector<NavState> states;
    std::vector<NavEdge> edges;


    // This can be optionally used to store the imu-sequences and preintegrations.
    std::shared_ptr<std::vector<std::pair<Preintegration, ImuSequence>>> local_imu_data;

    double weight_P = 10;
    double weight_V = 1;
    double weight_R = 100;

    double weight_change_a = 200;
    double weight_change_g = 1000;

    void Clear()
    {
        states.clear();
        edges.clear();
        local_imu_data = nullptr;
    }

    Vec3 WeightPVR() { return Vec3(weight_P, weight_V, weight_R); }


    void PreintAll()
    {
        for (auto& e : edges)
        {
            auto& s1 = states[e.from];

            *e.preint = Imu::Preintegration(s1.velocity_and_bias);
            e.preint->IntegrateMidPoint(*e.data, true);
        }
    }

    void SanityCheck()
    {
        for (auto& e : edges)
        {
            SAIGA_ASSERT(e.data->time_begin == states[e.from].time);
            SAIGA_ASSERT(e.data->time_end == states[e.to].time);
            SAIGA_ASSERT(e.data->time_end - e.data->time_begin > 0);
        }
    }

    void SolveCeres(const SolverOptions& params, bool ad = true);
    void Solve(const SolverOptions& params);


    void MakeRandom(int N, int K, double dt);

    double chi2() const;
    double chi2Print(double th) const;
};

SAIGA_VISION_API std::ostream& operator<<(std::ostream& strm, const DecoupledImuScene& scene);

}  // namespace Saiga::Imu
