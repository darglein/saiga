/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "Solver.h"
namespace Saiga::Imu
{
Vec3 SolveGlobalGyroBias(ArrayView<std::pair<const ImuSequence*, Quat> > data, int max_its)
{
    Vec3 current_bias = Vec3::Zero();
    for (int its = 0; its < max_its; ++its)
    {
        double chi2 = 0;
        Mat3 JtJ    = Mat3::Zero();
        Vec3 Jtb    = Vec3::Zero();
        for (int i = 1; i < data.size(); ++i)
        {
            auto& d1 = data[i - 1];
            auto& d2 = data[i];

            // the preintegration is from previous to current
            Imu::Preintegration preint(current_bias);
            preint.IntegrateForward(*d2.first);

            // 1. Compute residual
            Quat relative_rotation = d1.second.inverse() * d2.second;
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
    }
    return current_bias;
}

}  // namespace Saiga::Imu
