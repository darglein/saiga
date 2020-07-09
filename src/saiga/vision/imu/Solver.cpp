/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Solver.h"

#include "saiga/vision/util/Random.h"
namespace Saiga::Imu
{
Vec3 SolveGlobalGyroBias(ArrayView<ImuPosePair> data)
{
    Vec3 current_bias = Vec3::Zero();

    double chi2 = 0;
    Mat3 JtJ    = Mat3::Zero();
    Vec3 Jtb    = Vec3::Zero();
    for (int i = 0; i < data.size(); ++i)
    {
        auto& d = data[i];

        // the preintegration is from previous to current
        //        Imu::Preintegration preint(current_bias);
        //        preint.IntegrateMidPoint(*std::get<0>(d));

        auto& preint = *d.preint_12;

        //        std::cout << d.pose1->unit_quaternion() << std::endl;
        //        std::cout << d.pose2->unit_quaternion() << std::endl;

        // 1. Compute residual
        Quat relative_rotation = d.pose1->unit_quaternion().inverse() * d.pose2->unit_quaternion();
        Quat delta_rotation    = preint.delta_R.inverse() * relative_rotation;
        Vec3 residual          = Sophus::SO3d(delta_rotation).log();


        // 2. Compute Jacobian
        Mat3 Jlinv;
        Sophus::leftJacobianInvSO3(residual, Jlinv);
        Mat3 J = Jlinv * preint.J_R_Biasg;


        // 3. Scale by inverse time
        // Assuming additive gaussian noise,
        double sigma_inv = 1.0 / (preint.delta_t);
        J *= sigma_inv;
        residual *= sigma_inv;


        // 4. Add to JtJ and Jtb.
        chi2 += residual.squaredNorm();
        JtJ += J.transpose() * J;
        Jtb += J.transpose() * residual;
    }

    // 5. Solve and add delta to current estimate
    Vec3 x = JtJ.ldlt().solve(Jtb);
    current_bias += x;

    return current_bias;
}



std::pair<double, Vec3> SolveScaleGravityLinear(ArrayView<ImuPoseTriplet> data)
{
    int N = data.size() + 2;
    double scale_start;
    Vec3 gravity_start;

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
        //            Mat3 Rc3 = Twc3.unit_quaternion().matrix();

        // Stack to A/B matrix
        // lambda*s + beta*g = gamma
        Vec3 lambda = (pc2 - pc1) * dt23 + (pc2 - pc3) * dt12;
        Mat3 beta   = 0.5 * Mat3::Identity() * (dt12 * dt12 * dt23 + dt12 * dt23 * dt23);

        //            Vec3 gamma = (Rc3 - Rc2) * pcb * dt12 + (Rc1 - Rc2) * pcb * dt23 + Rc1 * Rcb * dp12 * dt23 -
        //                         Rc2 * Rcb * dp23 * dt12 - Rc1 * Rcb * dv12 * dt12 * dt23;

        Vec3 gamma = Rc1 * dp12 * dt23 - Rc2 * dp23 * dt12 - Rc1 * dv12 * dt12 * dt23;
        //            Vec3 gamma = Vec3::Zero();


        A.block<3, 1>(3 * i, 0) = lambda * weight;
        A.block<3, 3>(3 * i, 1) = beta * weight;
        B.block<3, 1>(3 * i, 0) = gamma * weight;
    }


    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinV | Eigen::ComputeThinU);
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd V = svd.matrixV();

    //    std::cout << A << std::endl;
    //    std::cout << B << std::endl;

    //    exit(0);
    SAIGA_ASSERT(U.cols() == 4);

    Eigen::DiagonalMatrix<double, 4> S;
    S.diagonal() = svd.singularValues();

    double condition_number = S.diagonal()(0) / S.diagonal()(3);
    std::cout << "c = " << condition_number << std::endl;

    // Then x = vt'*winv*u'*B
    Vec4 x        = V * S.inverse() * U.transpose() * B;
    scale_start   = x(0);
    gravity_start = x.segment<3>(1);

    return {scale_start, gravity_start};
}



std::tuple<double, Vec3, Vec3> SolveScaleGravityBiasLinear(ArrayView<ImuPoseTriplet> data, const Vec3& gravity_estimate)
{
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

        double weight = 1.0 / (dt12 + dt23) * triplet.weight;

        //        double weight = 1;

        // Position of camera center
        Vec3 pc1 = Twc1.translation();
        Vec3 pc2 = Twc2.translation();
        Vec3 pc3 = Twc3.translation();

        // Rotation of camera, Rwc
        Mat3 Rc1 = Twc1.unit_quaternion().matrix();
        Mat3 Rc2 = Twc2.unit_quaternion().matrix();
        //        Mat3 Rc3 = Twc3.unit_quaternion().matrix();


        // Stack to C/D matrix
        // lambda*s + phi*dthetaxy + zeta*ba = psi
        Vec3 lambda = (pc2 - pc1) * dt23 + (pc2 - pc3) * dt12;

        Mat3 phi = -0.5 * (dt12 * dt12 * dt23 + dt12 * dt23 * dt23) * Rwi *
                   skew(GI);  // note: this has a '-', different to paper

        //            Mat3 zeta = Rc2 * Rcb * Jpba23 * dt12 + Rc1 * Rcb * Jvba12 * dt12 * dt23 - Rc1 * Rcb * Jpba12
        //            * dt23;
        Mat3 zeta = Rc2 * Jpba23 * dt12 + Rc1 * Jvba12 * dt12 * dt23 - Rc1 * Jpba12 * dt23;

        //            Vec3 psi = Rc1 * Rcb * dp12 * dt23 - (Rc2 - Rc3) * pcb * dt12 - Rc2 * Rcb * dp23 * dt12 -
        //                       Rc1 * Rcb * dv12 * dt23 * dt12 -
        //                       0.5 * Rwi * GI * (dt12 * dt12 * dt23 + dt12 * dt23 * dt23);

        Vec3 psi = Rc1 * dp12 * dt23 - Rc2 * dp23 * dt12 - Rc1 * dv12 * dt23 * dt12 -
                   0.5 * Rwi * GI * (dt12 * dt12 * dt23 + dt12 * dt23 * dt23);

        C.block<3, 1>(3 * i, 0) = lambda * weight;
        C.block<3, 2>(3 * i, 1) = phi.block<3, 2>(0, 0) * weight;
        C.block<3, 3>(3 * i, 3) = zeta * weight;
        D.block<3, 1>(3 * i, 0) = psi * weight;
    }

    Eigen::JacobiSVD<Eigen::MatrixXd> svd(C, Eigen::ComputeThinV | Eigen::ComputeThinU);
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd V = svd.matrixV();

    Eigen::DiagonalMatrix<double, 6> S;
    S.diagonal() = svd.singularValues();

    double condition_number = S.diagonal()(0) / S.diagonal()(5);
    std::cout << "c = " << condition_number << std::endl;


    // Then y = vt'*winv*u'*D
    Eigen::MatrixXd y = V * S.inverse() * U.transpose() * D;
    double scale      = y(0);


    Vec2 dthetaxy = y.block<2, 1>(1, 0);
    Vec3 bias_acc = y.block<3, 1>(3, 0);


    // dtheta = [dx;dy;0]
    Vec3 dtheta(dthetaxy(0), dthetaxy(1), 0);
    Mat3 Rwi_    = Rwi * Sophus::SO3d::exp(dtheta).matrix();
    Vec3 gravity = Rwi_ * GI;

    return {scale, gravity, bias_acc};
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
