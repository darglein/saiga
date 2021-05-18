/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "Solver.h"

#include "saiga/vision/kernels/Robust.h"
#include "saiga/vision/util/Random.h"
namespace Saiga::Imu
{
std::pair<Vec3, double> SolveGlobalGyroBias(ArrayView<ImuPosePair> data, double huber_threshold)
{
    double chi2 = 0;
    Mat3 JtJ    = Mat3::Zero();
    Vec3 Jtb    = Vec3::Zero();
    for (int i = 0; i < data.size(); ++i)
    {
        ImuPosePair& d = data[i];

        // the preintegration is from previous to current
        const Preintegration& preint = *d.preint_12;

        Mat3 J;
        Vec3 residual = preint.RotationalError(d.pose1->so3(), d.pose2->so3(), &J);

        double res_2       = residual.squaredNorm();
        d.chi2_residual    = res_2;
        auto rw            = Kernel::CauchyLoss<double>(huber_threshold, res_2);
        res_2              = rw(0);
        double loss_weight = rw(1);

        // 4. Add to JtJ and Jtb.
        chi2 += res_2;
        JtJ += loss_weight * (J.transpose() * J);
        Jtb -= loss_weight * (J.transpose() * residual);
    }

    // 5. Solve and add delta to current estimate
    Vec3 x      = JtJ.ldlt().solve(Jtb);
    double rmse = sqrt(chi2 / data.size());

    return {x, rmse};
}



std::pair<double, Vec3> SolveScaleGravityLinear(ArrayView<ImuPoseTriplet> data, const SE3& pose_to_imu)
{
    int N = data.size() + 2;
    double scale_start;
    Vec3 gravity_start;


    Mat3 Rcb = pose_to_imu.unit_quaternion().matrix();
    Vec3 pcb = pose_to_imu.translation();


    Eigen::MatrixXd A(3 * (N - 2), 4);
    Eigen::MatrixXd B(3 * (N - 2), 1);

    for (int i = 0; i < N - 2; ++i)
    {
        auto triplet = data[i];

        auto dt12 = triplet.preint_12->delta_t;
        auto dt23 = triplet.preint_23->delta_t;

        // Pre-integrated measurements
        Vec3 dp12 = triplet.preint_12->delta_x;
        Vec3 dv12 = triplet.preint_12->delta_v;
        Vec3 dp23 = triplet.preint_23->delta_x;

        SE3 Twc1 = *triplet.pose1;
        SE3 Twc2 = *triplet.pose2;
        SE3 Twc3 = *triplet.pose3;

        double weight = 1.0 / (dt12 + dt23) * triplet.weight;

        // Position of camera center
        Vec3 pc1 = Twc1.translation();
        Vec3 pc2 = Twc2.translation();
        Vec3 pc3 = Twc3.translation();

        // Rotation of camera, Rwc
        Mat3 Rc1 = Twc1.unit_quaternion().matrix();
        Mat3 Rc2 = Twc2.unit_quaternion().matrix();
        Mat3 Rc3 = Twc3.unit_quaternion().matrix();

        // Stack to A/B matrix
        // lambda*s + beta*g = gamma
        Vec3 lambda = (pc2 - pc1) * dt23 + (pc2 - pc3) * dt12;
        Mat3 beta   = 0.5 * Mat3::Identity() * (dt12 * dt12 * dt23 + dt12 * dt23 * dt23);

#if 0
        Vec3 gamma = Rc1 * dp12 * dt23 - Rc2 * dp23 * dt12 - Rc1 * dv12 * dt12 * dt23;
#else
        Vec3 gamma = (Rc3 - Rc2) * pcb * dt12 + (Rc1 - Rc2) * pcb * dt23 + Rc1 * Rcb * dp12 * dt23 -
                     Rc2 * Rcb * dp23 * dt12 - Rc1 * Rcb * dv12 * dt12 * dt23;
#endif


        A.block<3, 1>(3 * i, 0) = lambda * weight;
        A.block<3, 3>(3 * i, 1) = beta * weight;
        B.block<3, 1>(3 * i, 0) = gamma * weight;
    }


    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinV | Eigen::ComputeThinU);


    //    Eigen::DiagonalMatrix<double, 4> S;
    //    S.diagonal() = svd.singularValues();
    //    double condition_number = S.diagonal()(0) / S.diagonal()(3);
    //    std::cout << "c = " << condition_number << std::endl;

    Vec4 x = svd.solve(B);

    scale_start   = x(0);
    gravity_start = x.segment<3>(1);

    return {scale_start, gravity_start};
}



std::tuple<double, Vec3, Vec3, double> SolveScaleGravityBiasLinear(ArrayView<ImuPoseTriplet> data,
                                                                   const Vec3& gravity_estimate, const SE3& pose_to_imu)
{
    Mat3 Rcb = pose_to_imu.unit_quaternion().matrix();
    Vec3 pcb = pose_to_imu.translation();

    int N = data.size() + 2;
    // Step 3.
    // Use gravity magnitude 9.8 as constraint
    // gI = [0;0;1], the normalized gravity vector in an inertial frame, NED type
    // with no orientation.
    Vec3 gI(0, 0, 1);

    // Normalized approx. gravity vecotr in world frame
    Vec3 gwn = gravity_estimate.normalized();
    // Debug log
    // cout<<"gw normalized: "<<gwn<<endl;

    // vhat = (gI x gw) / |gI x gw|
    Vec3 gIxgwn       = gI.cross(gwn);
    double normgIxgwn = gIxgwn.norm();
    Vec3 vhat         = gIxgwn / normgIxgwn;
    double theta      = std::atan2(normgIxgwn, gI.dot(gwn));


    double chi2 = 0;

    Mat3 Rwi = Sophus::SO3d::exp(vhat * theta).matrix();

    // vi orbslam:       9.81
    // standard gravity: 9.806
    Vec3 GI = gI * 9.81;

    // Solve C*x=D for x=[s,dthetaxy,ba] (1+2+3)x1 vector
    Eigen::MatrixXd C(3 * (N - 2), 6);
    Eigen::MatrixXd D(3 * (N - 2), 1);

    C.setZero();
    D.setZero();

    for (int i = 0; i < N - 2; ++i)
    {
        auto triplet = data[i];

        auto dt12 = triplet.preint_12->delta_t;
        auto dt23 = triplet.preint_23->delta_t;

        // Pre-integrated measurements
        Vec3 dp12 = triplet.preint_12->delta_x;
        Vec3 dv12 = triplet.preint_12->delta_v;
        Vec3 dp23 = triplet.preint_23->delta_x;

        Mat3 Jpba12 = triplet.preint_12->J_P_Biasa;
        Mat3 Jvba12 = triplet.preint_12->J_V_Biasa;
        Mat3 Jpba23 = triplet.preint_23->J_P_Biasa;

        SE3 Twc1 = *triplet.pose1;
        SE3 Twc2 = *triplet.pose2;
        SE3 Twc3 = *triplet.pose3;

        double weight = 100.0 / (dt12 + dt23) * triplet.weight;

        //        double weight = 1;

        // Position of camera center
        Vec3 pc1 = Twc1.translation();
        Vec3 pc2 = Twc2.translation();
        Vec3 pc3 = Twc3.translation();

        // Rotation of camera, Rwc
        Mat3 Rc1 = Twc1.unit_quaternion().matrix();
        Mat3 Rc2 = Twc2.unit_quaternion().matrix();
        Mat3 Rc3 = Twc3.unit_quaternion().matrix();


        // Stack to C/D matrix
        // lambda*s + phi*dthetaxy + zeta*ba = psi
        Vec3 lambda = (pc2 - pc1) * dt23 + (pc2 - pc3) * dt12;
        Mat3 phi    = -0.5 * (dt12 * dt12 * dt23 + dt12 * dt23 * dt23) * Rwi * skew(GI);

#if 0
        Mat3 zeta = Rc2 * Jpba23 * dt12 + Rc1 * Jvba12 * dt12 * dt23 - Rc1 * Jpba12 * dt23;
        Vec3 psi = Rc1 * dp12 * dt23 - Rc2 * dp23 * dt12 - Rc1 * dv12 * dt23 * dt12 -
                   0.5 * Rwi * GI * (dt12 * dt12 * dt23 + dt12 * dt23 * dt23);
#else
        Mat3 zeta = Rc2 * Rcb * Jpba23 * dt12 + Rc1 * Rcb * Jvba12 * dt12 * dt23 - Rc1 * Rcb * Jpba12 * dt23;

        Vec3 psi = Rc1 * Rcb * dp12 * dt23 - (Rc2 - Rc3) * pcb * dt12 - Rc2 * Rcb * dp23 * dt12 -
                   Rc1 * Rcb * dv12 * dt23 * dt12 - 0.5 * Rwi * GI * (dt12 * dt12 * dt23 + dt12 * dt23 * dt23);
#endif


        Vec3 residual = (lambda - psi) * weight;

        //        auto rw = Kernel::CauchyLoss(0.3, residual.squaredNorm());
        //        weight *= sqrt(rw(1));
        //        chi2 += rw(0);
        chi2 += residual.squaredNorm();



        //        chi2 += residual.squaredNorm();
        //        std::cout << i << " " << residual.norm() << std::endl;

        C.block<3, 1>(3 * i, 0) = lambda * weight;
        C.block<3, 2>(3 * i, 1) = phi.block<3, 2>(0, 0) * weight;
        C.block<3, 3>(3 * i, 3) = zeta * weight;
        D.block<3, 1>(3 * i, 0) = psi * weight;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(C, Eigen::ComputeThinV | Eigen::ComputeThinU);

    //    Eigen::DiagonalMatrix<double, 6> S;
    //    S.diagonal() = svd.singularValues();
    //    double condition_number = S.diagonal()(0) / S.diagonal()(5);
    //    std::cout << "c = " << condition_number << std::endl;


    Vec6 y       = svd.solve(D);
    double scale = y(0);


    Vec2 dthetaxy = y.segment<2>(1);
    Vec3 bias_acc = y.segment<3>(3);


    // dtheta = [dx;dy;0]
    Vec3 dtheta(dthetaxy(0), dthetaxy(1), 0);
    Mat3 Rwi_    = Rwi * Sophus::SO3d::exp(dtheta).matrix();
    Vec3 gravity = Rwi_ * GI;

    double rmse = sqrt(chi2 / N);
    //    std::cout << "RMSE: " << rmse << std::endl;

    return {scale, gravity, bias_acc, rmse};
}

std::vector<Synthetic::State> Synthetic::GenerateStates(int N, double dt, double sigma_angular_acceleration,
                                                        double sigma_linear_acceleration)
{
    std::vector<Synthetic::State> states;

    //    State initial_state;
    //    initial_state.pose = Random::randomSE3();
    //    initial_state.angular_acceleration;
    return states;
}

}  // namespace Saiga::Imu
