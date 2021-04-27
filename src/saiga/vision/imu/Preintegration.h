/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionTypes.h"

#include "Imu.h"

#include <vector>


namespace Saiga::Imu
{
struct SAIGA_VISION_API Preintegration
{
    Preintegration(const Vec3& bias_gyro = Vec3::Zero(), const Vec3& bias_accel = Vec3::Zero(), double cov_gyro = 0,
                   double cov_acc = 0)
        : bias_gyro_lin(bias_gyro), bias_accel_lin(bias_accel)
#if 0
          , cov_gyro(cov_gyro), cov_acc(cov_acc)
#endif
    {
    }
    Preintegration(const VelocityAndBias& vb) : Preintegration(vb.gyro_bias, vb.acc_bias) {}

    void Add(const Data& data, double dt, bool derive) { Add(data.omega, data.acceleration, dt, derive); }

    // The main integration function.
    // This assumes a constant w and a for dt time.
    void Add(const Vec3& omega_with_bias, const Vec3& acc_with_bias, double dt, bool derive);

    // Adds the complete sequence using forward integration (explicit Euler).
    void IntegrateForward(const ImuSequence& sequence, bool derive);

    void IntegrateMidPoint(const ImuSequence& sequence, bool derive);



    // Predicts the state after this sequence.
    //
    std::pair<SE3, Vec3> Predict(const SE3& initial_pose, const Vec3& initial_velocity, const Vec3& g) const;



    // Imu error for VSLAM.
    // The poses must be given in body(IMU) space.
    Vec9 ImuError(const VelocityAndBias& delta_bias_i, const Vec3& v_i, const SE3& _pose_i, const Vec3& v_j,
                  const SE3& _pose_j, const Gravity& g, double scale, const Vec3& weight_pvr,
                  Matrix<double, 9, 3>* J_biasa = nullptr, Matrix<double, 9, 3>* J_biasg = nullptr,
                  Matrix<double, 9, 3>* J_v1 = nullptr, Matrix<double, 9, 3>* J_v2 = nullptr,
                  Matrix<double, 9, 6>* J_p1 = nullptr, Matrix<double, 9, 6>* J_p2 = nullptr,
                  Matrix<double, 9, 1>* J_scale = nullptr, Matrix<double, 9, 3>* J_g = nullptr) const;


    Vec6 BiasChangeError(const VelocityAndBias& bias_i, const VelocityAndBias& delta_bias_i,
                         const VelocityAndBias& bias_j, const VelocityAndBias& delta_bias_j, double weight_acc,
                         double weight_gyro, Matrix<double, 6, 6>* J_a_g_i = nullptr,
                         Matrix<double, 6, 6>* J_a_g_j = nullptr) const;


    Vec3 RotationalError(const SO3& pose_i, const SO3& pose_j, Matrix<double, 3, 3>* J_g = nullptr) const;

    // Integrated values (Initialized to identity/0);
    double delta_t = 0;
    SO3 delta_R;
    Vec3 delta_x = Vec3::Zero();
    Vec3 delta_v = Vec3::Zero();

    // Derivative w.r.t. the gyro bias
    Mat3 J_R_Biasg = Mat3::Zero();
    Mat3 J_P_Biasg = Mat3::Zero();
    Mat3 J_V_Biasg = Mat3::Zero();


    // Derivative w.r.t. the acc bias
    Mat3 J_P_Biasa = Mat3::Zero();
    Mat3 J_V_Biasa = Mat3::Zero();

#if 0
    // noise covariance propagation of delta measurements
        Eigen::Matrix<double, 9, 9> cov_P_V_Phi;
    double cov_gyro = 0;
    double cov_acc  = 0;
#endif

    Vec3 GetBiasAcc() { return bias_accel_lin; }
    Vec3 GetBiasGyro() { return bias_gyro_lin; }

   private:
    // Linear bias, which is subtracted from the meassurements.
    // Private because changing the bias invalidates the preintegration.
    Vec3 bias_gyro_lin, bias_accel_lin;
};


}  // namespace Saiga::Imu
