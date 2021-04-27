/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "DecoupledImuScene.h"

#include "saiga/vision/util/Random.h"

namespace Saiga::Imu
{
void DecoupledImuScene::MakeRandom(int N, int K, double dt)
{
    auto data = GenerateRandomSequence(N, K, dt);


    weight_change_a = Random::sampleDouble(0.5, 2);
    weight_change_g = Random::sampleDouble(0.5, 2);



    weight_P = Random::sampleDouble(0.5, 2);
    weight_V = Random::sampleDouble(0.5, 2);
    weight_R = Random::sampleDouble(0.5, 2);


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
        e.preint->IntegrateForward(*e.data, true);


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
    }

    for (int i = 0; i < N - 1; ++i)
    {
        auto& s1      = states[i];
        auto& s2      = states[i + 1];
        auto& e       = edges[i];
        e.weight_bias = Vec2::Random();
        e.weight_pvr  = Random::sampleDouble(0.5, 1.9);
        //        e.weight_bias = 10;
        //        e.weight_pvr  = 10;
        e.data->AddBias(s1.velocity_and_bias.gyro_bias, s1.velocity_and_bias.acc_bias);
        e.data->AddGravity(s1.velocity_and_bias.gyro_bias, s1.pose.so3(), -gravity.Get());

        s1.time = e.data->time_begin;
        s2.time = e.data->time_end;
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

        auto vb = s1.velocity_and_bias;

        vb.acc_bias += s1.delta_bias.acc_bias;
        vb.gyro_bias += s1.delta_bias.gyro_bias;

        Imu::Preintegration preint(vb);
        preint.IntegrateMidPoint(*e.data, false);

        Vec9 residual =
            preint.ImuError(VelocityAndBias(), s1.velocity_and_bias.velocity, s1.pose, s2.velocity_and_bias.velocity,
                            s2.pose, gravity, scale, Vec3(weight_P, weight_V, weight_R) * e.weight_pvr);



        Vec6 res_bias_change =
            e.preint->BiasChangeError(s1.velocity_and_bias, s1.delta_bias, s2.velocity_and_bias, s2.delta_bias,
                                      weight_change_a * e.weight_bias(0), weight_change_g * e.weight_bias(1));



        sum += residual.squaredNorm() + res_bias_change.squaredNorm();
    }
    return sum;
}

double DecoupledImuScene::chi2Print(double th) const
{
    double sum = 0;
    for (auto& e : edges)
    {
        auto& s1 = states[e.from];
        auto& s2 = states[e.to];

        auto vb = s1.velocity_and_bias;

        vb.acc_bias += s1.delta_bias.acc_bias;
        vb.gyro_bias += s1.delta_bias.gyro_bias;

        Imu::Preintegration preint(vb);
        preint.IntegrateMidPoint(*e.data, false);

        SAIGA_ASSERT(e.preint);
        SAIGA_ASSERT(e.data);

        Vec9 residual =
            preint.ImuError(VelocityAndBias(), s1.velocity_and_bias.velocity, s1.pose, s2.velocity_and_bias.velocity,
                            s2.pose, gravity, scale, Vec3(weight_P, weight_V, weight_R) * e.weight_pvr);



        Vec6 res_bias_change =
            e.preint->BiasChangeError(s1.velocity_and_bias, s1.delta_bias, s2.velocity_and_bias, s2.delta_bias,
                                      weight_change_a * e.weight_bias(0), weight_change_g * e.weight_bias(1));
        double r = residual.squaredNorm() + res_bias_change.squaredNorm();

        if (!std::isfinite(r) || r > th)
        {
            std::cout << "Edge " << e.from << " -> " << e.to << " dt: " << preint.delta_t
                      << " Res: " << residual.squaredNorm() << std::endl;

            std::cout << s1.velocity_and_bias.acc_bias.transpose() << " " << s1.velocity_and_bias.gyro_bias.transpose()
                      << std::endl;
            std::cout << s2.velocity_and_bias.acc_bias.transpose() << " " << s2.velocity_and_bias.gyro_bias.transpose()
                      << std::endl;

            std::cout << s1.delta_bias.acc_bias.transpose() << " " << s1.delta_bias.gyro_bias.transpose() << std::endl;
            std::cout << s2.delta_bias.acc_bias.transpose() << " " << s2.delta_bias.gyro_bias.transpose() << std::endl;

            std::cout << weight_change_a << " " << weight_change_g << std::endl;
            std::cout << e.weight_bias.transpose() << std::endl;
            std::cout << residual.transpose() << std::endl;
            std::cout << res_bias_change.transpose() << std::endl;
            std::cout << std::endl;
        }



        sum += r;
    }
    return sum;
}

std::ostream& operator<<(std::ostream& strm, const DecoupledImuScene& scene)
{
    strm << "[Imu Scene]" << std::endl;
    strm << " States: " << scene.states.size() << std::endl;
    strm << " Edges: " << scene.edges.size() << std::endl;
    strm << " Chi2: " << scene.chi2() << std::endl;


    for (auto& s : scene.states)
    {
        std::cout << "> State " << s.time << ": " << s.velocity_and_bias.gyro_bias.transpose() << " | "
                  << s.velocity_and_bias.acc_bias.transpose() << std::endl;
    }

    //    for (auto& e : scene.edges)
    //    {
    //        std::cout << *e.data << std::endl;
    //    }
    return strm;
}
}  // namespace Saiga::Imu
