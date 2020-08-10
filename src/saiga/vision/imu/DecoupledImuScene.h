/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/imu/all.h"


namespace Saiga::Imu
{
struct SAIGA_VISION_API NavState
{
    SE3 pose;
    Imu::VelocityAndBias velocity_and_bias;
    double time;
};

struct SAIGA_VISION_API NavEdge
{
    Preintegration* preint = nullptr;
    ImuSequence* data      = nullptr;
    int from, to;
};

class SAIGA_VISION_API DecoupledImuScene
{
   public:
    struct SolverOptions
    {  // Solver options
        bool solve_bias_gyro = true;
        bool solve_bias_acc  = true;
        bool solve_velocity  = true;
        bool solve_gravity   = true;
        bool solve_scale     = false;

        bool use_global_bias = false;
        int max_its          = 10;
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



    void SanityCheck()
    {
        for (auto& e : edges)
        {
            SAIGA_ASSERT(e.data->time_begin == states[e.from].time);
            SAIGA_ASSERT(e.data->time_end == states[e.to].time);
        }
    }

    void SolveCeres(const SolverOptions& params, bool ad = true);

    void MakeRandom(int N, int K, double dt);

    double chi2() const;
};

SAIGA_VISION_API std::ostream& operator<<(std::ostream& strm, const DecoupledImuScene& scene);

}  // namespace Saiga::Imu
