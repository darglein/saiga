/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionTypes.h"

#include <vector>


namespace Saiga::Imu
{
using OmegaType        = Vec3;
using AccelerationType = Vec3;

template <typename T>
using VelocityBase = Vector<T, 3>;
using Velocity     = VelocityBase<double>;

template <typename T>
using GyroBiasBase = Vector<T, 3>;
using GyroBias     = GyroBiasBase<double>;

template <typename T>
using AccBiasBase = Vector<T, 3>;
using AccBias     = AccBiasBase<double>;

// (Linear) velocity and ImuBias stacked as
// [Velocity | GyroBias | AccBias]
template <typename T>
struct VelocityAndBiasBase
{
    VelocityBase<T> velocity  = VelocityBase<T>::Zero();
    GyroBiasBase<T> gyro_bias = GyroBiasBase<T>::Zero();
    AccBiasBase<T> acc_bias   = AccBiasBase<T>::Zero();

    template <typename G>
    VelocityAndBiasBase<G> cast()
    {
        VelocityAndBiasBase<G> result;
        result.velocity  = velocity.template cast<G>();
        result.gyro_bias = gyro_bias.template cast<G>();
        result.acc_bias  = acc_bias.template cast<G>();
        return result;
    }
};
using VelocityAndBias = VelocityAndBiasBase<double>;


// The angular velocity (omega) is a combined angle-axis representation.
// The length of omega is the angle and the direction is the axis.
template <typename T>
Eigen::Quaternion<T> OmegaToQuaternion(Vector<T, 3> omega, T dt)
{
    T angle = omega.norm();
    if (angle < std::numeric_limits<T>::epsilon())
    {
        return Eigen::Quaternion<T>::Identity();
    }

    auto axis = omega / angle;
    Eigen::AngleAxis<T> aa(angle * dt, axis);
    Eigen::Quaternion<T> q(aa);
    return q;

#if 0
    T theta_half      = omega.norm() * 0.5 * dt;
    T sinc_theta_half = sinc(theta_half);
    T cos_theta_half  = cos(theta_half);
    Eigen::Quaternion<T> dq;
    dq.vec() = sinc_theta_half * omega * 0.5 * dt;
    dq.w()   = cos_theta_half;
    std::cout << dq << std::endl;
    return dq;
#endif
}

template <typename T>
Vector<T, 3> QuaternionToOmega(Eigen::Quaternion<T> q)
{
    Eigen::AngleAxis<T> aa(q);
    return aa.axis() * aa.angle();
}


struct SAIGA_VISION_API Data
{
    // Angular velocity in [rad/s] given by the gyroscope.
    OmegaType omega;

    // Linear acceleration in [m/s^2] given my the accelerometer.
    AccelerationType acceleration;

    // Timestamp of the sensor reading in [s]
    double timestamp = std::numeric_limits<double>::infinity();

    Data() = default;
    Data(const Vec3& omega, const Vec3& acceleration, double timestamp)
        : omega(omega), acceleration(acceleration), timestamp(timestamp)
    {
    }

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW


    static Data Interpolate(const Data& before, const Data& after, double new_timestamp)
    {
        // don't extrapolate
        SAIGA_ASSERT(before.timestamp <= new_timestamp);
        SAIGA_ASSERT(after.timestamp >= new_timestamp);
        double alpha = (new_timestamp - before.timestamp) / (after.timestamp - before.timestamp);


        Data result;
        result.omega        = (1.0 - alpha) * before.omega + alpha * after.omega;
        result.acceleration = (1.0 - alpha) * before.acceleration + alpha * after.acceleration;
        result.timestamp    = new_timestamp;
        return result;
    }
};

SAIGA_VISION_API std::ostream& operator<<(std::ostream& strm, const Imu::Data& data);


struct Sensor
{
    // Number of meassueremnt per second in [hz]
    double frequency;

    // Standard deviation of the sensor noise
    double acceleration_sigma;
    double acceleration_random_walk;

    double omega_sigma;
    double omega_random_walk;

    // Transformation from the sensor coordinate system to the devices' coordinate system.
    SE3 sensor_to_body;
};

SAIGA_VISION_API std::ostream& operator<<(std::ostream& strm, const Imu::Sensor& sensor);


// Camera frames appear in lower frequency than IMU meassurements. This structs is a collection of all IMU meassurements
// between two camera frames. If this Frame objects belongs to camera frame i, then it contains all IMU data from frame
// (i-1) to frame i. For sanity, the timestamp of this camera frame should be larger than all the IMU datas.
//
// This struct also contains an interpolated IMU meassurement exactly at the image time. To compute this interpolated
// value, one IMU meassurement after the image is required. Implementations of live sensors should therefore return the
// image only after one more IMU input has arrived.
struct SAIGA_VISION_API Frame
{
    // Timestamp of the image.
    double timestamp = std::numeric_limits<double>::infinity();

    // All meassurements since the last frame. The frame 0 contains all meassurements since the beginning. All elements
    // in this array should have a smaller timestamp than 'timestamp' above.
    std::vector<Imu::Data> imu_data_since_last_frame;

    // The clostest imu data right after this image frame. The timestamp difference between this and 'timestamp' from
    // above should be less than the imu frequency.
    Imu::Data imu_directly_after_this_frame;


    Imu::Data interpolated_imu;

    void computeInterplatedImuValue();

    void sanityCheck(const Sensor& sensor);
};


}  // namespace Saiga::Imu
