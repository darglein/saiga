/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "DecoupledImuScene.h"

#include "saiga/vision/ceres/CeresHelper.h"
#include "saiga/vision/imu/CeresPreintegration.h"
#include "saiga/vision/util/Random.h"

#include "ceres/autodiff_cost_function.h"
#include "ceres/local_parameterization.h"
namespace Saiga::Imu
{
void DecoupledImuScene::MakeRandom(int N, int K, double dt)
{
    auto data = GenerateRandomSequence(N, K, dt);



    states.resize(N);
    edges.resize(N - 1);

    states.front().pose                       = Random::randomSE3();
    states.front().velocity_and_bias.velocity = Vec3::Random();

    local_imu_data = std::make_shared<std::vector<std::pair<Preintegration, ImuSequence>>>();
    local_imu_data->resize(edges.size());


    // Integrate to get actual poses
    for (int i = 0; i < N - 1; ++i)
    {
        auto& s1 = states[i];
        auto& s2 = states[i + 1];
        auto& e  = edges[i];

        e.from   = i;
        e.to     = i + 1;
        e.preint = &(*local_imu_data)[i].first;
        e.data   = &(*local_imu_data)[i].second;

        (*e.data) = data[i + 1];
        *e.preint = Imu::Preintegration();
        e.preint->IntegrateForward(*e.data);


        auto pose_v = e.preint->Predict(s1.pose, s1.velocity_and_bias.velocity, Vec3::Zero());

        s2.pose                       = pose_v.first;
        s2.velocity_and_bias.velocity = pose_v.second;
    }



    gravity.R = Random::randomSE3().so3();

    states.front().velocity_and_bias.acc_bias  = Vec3::Random() * 0.1;
    states.front().velocity_and_bias.gyro_bias = Vec3::Random() * 0.1;

    // Add bias+noise to measurements
    for (int i = 1; i < N; ++i)
    {
        auto& s0 = states[i - 1];
        auto& s1 = states[i];


        s1.velocity_and_bias.acc_bias  = s0.velocity_and_bias.acc_bias;
        s1.velocity_and_bias.gyro_bias = s0.velocity_and_bias.gyro_bias;

        //        s1.velocity_and_bias.acc_bias += Vec3::Random() * 0.0001;
        //        s1.velocity_and_bias.gyro_bias += Vec3::Random() * 0.0001;
    }

    for (int i = 0; i < N - 1; ++i)
    {
        auto& s1 = states[i];
        auto& e  = edges[i];
        e.data->AddBias(s1.velocity_and_bias.gyro_bias, s1.velocity_and_bias.acc_bias);
        e.data->AddGravity(s1.velocity_and_bias.gyro_bias, s1.pose.so3(), -gravity.Get());
    }

    //        e.data->AddNoise(0.1, 0.1);
}

double DecoupledImuScene::chi2() const
{
    double sum = 0;
    for (auto& e : edges)
    {
        auto& s1 = states[e.from];
        auto& s2 = states[e.to];

        Imu::Preintegration preint(s1.velocity_and_bias);
        preint.IntegrateMidPoint(*e.data);

        Vec9 residual =
            preint.ImuError(VelocityAndBias(), s1.velocity_and_bias.velocity, s1.pose, s2.velocity_and_bias.velocity,
                            s2.pose, gravity, scale, Vec3(weight_P, weight_V, weight_R));


        sum += residual.squaredNorm();
    }
    return sum;
}

std::ostream& operator<<(std::ostream& strm, const DecoupledImuScene& scene)
{
    strm << "[Imu Scene]" << std::endl;
    strm << " States: " << scene.states.size() << std::endl;
    strm << " Edges: " << scene.edges.size() << std::endl;
    strm << " Chi2: " << scene.chi2() << std::endl;

    //    for (auto& e : scene.edges)
    //    {
    //        std::cout << *e.data << std::endl;
    //    }
    return strm;
}
}  // namespace Saiga::Imu
