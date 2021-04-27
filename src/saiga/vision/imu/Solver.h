/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/imu/all.h"


namespace Saiga::Imu
{
struct ImuPosePair
{
    const SE3* pose1;
    const SE3* pose2;
    const Imu::Preintegration* preint_12;

    // Residual squared norm (before robust cost function). Written by the solver.
    double chi2_residual = 0;
};

// Computes a global gyro bias which minimizes the relative rotational error.
// The input is a array of IMU Sequences and the global start and end rotation for that sequence.
//
// Notes:
//   - In a perfect world, the problem is linear, but usually 2 iterations are recommended.
//   - If you're computing the bias for a VI system, make sure to transform the camera frame to the IMU frame.
SAIGA_VISION_API std::pair<Vec3, double> SolveGlobalGyroBias(ArrayView<ImuPosePair> data, double huber_threshold = 1);



struct ImuPoseTriplet
{
    // Estimated IMU poses for example from visual odometry.
    // Make sure they are in IMU space!
    const SE3* pose1;
    const SE3* pose2;
    const SE3* pose3;

    // Preintegration from 1 to 2 and from 2 to 3.
    const Imu::Preintegration* preint_12;
    const Imu::Preintegration* preint_23;

    double weight = 1.0;
};

// pose_to_imu is the right-multiplicative transformation from the input space to imu space.
SAIGA_VISION_API std::pair<double, Vec3> SolveScaleGravityLinear(ArrayView<ImuPoseTriplet> data,
                                                                 const SE3& pose_to_imu);

SAIGA_VISION_API std::tuple<double, Vec3, Vec3, double> SolveScaleGravityBiasLinear(ArrayView<ImuPoseTriplet> data,
                                                                                    const Vec3& gravity_estimate,
                                                                                    const SE3& pose_to_imu);


namespace Synthetic
{
// Procedural Pose-IMU data for testing.
struct State
{
    double time;

    // The integrated pose
    SE3 pose;

    // Angular and linear velocity
    Vec3 omega;
    Vec3 velocity;

    Vec3 angular_acceleration;
    Vec3 linear_acceleration;
};

std::vector<State> GenerateStates(int N, double dt, double sigma_angular_acceleration,
                                  double sigma_linear_acceleration);

}  // namespace Synthetic


}  // namespace Saiga::Imu
