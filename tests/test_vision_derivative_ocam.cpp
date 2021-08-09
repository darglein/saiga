/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "saiga/vision/cameraModel/OCam.h"

#include "gtest/gtest.h"

#include "compare_numbers.h"
#include "numeric_derivative.h"

namespace Saiga
{
TEST(Derivative, ProjectOCam)
{
    Random::setSeed(903476346);
    srand(976157);

    OCam<double> cam;
    cam.SetAffineParams(Vec5::Random());
    std::vector<double> world_2_cam = {2185.330078, 1374.650024, -195.684006, -277.934998, 140.244003, 489.372009,
                                       -103.584000, -667.950012, 56.145100,   756.213989,  105.819000, -574.442017,
                                       -214.572998, 240.039001,  152.841995,  -27.509800,  -39.366600, -7.958420};

    Vec3 p = Random::MatrixUniform<Vec3>(-1, 1);

    Matrix<double, 2, 3> J_point_1, J_point_2;
    J_point_1.setZero();
    J_point_2.setZero();

    Matrix<double, 2, 5> J_affine_1, J_affine_2;
    J_affine_1.setZero();
    J_affine_2.setZero();


    Vec2 res1, res2;
    res1 = ProjectOCam<double>(p, cam.AffineParams(), world_2_cam, 0.5, &J_point_1, &J_affine_1).head<2>();

    std::cout << "res " << res1.transpose() << std::endl;

    {
        res2 = EvaluateNumeric(
            [=](auto p) {
                return ProjectOCam<double>(p, cam.AffineParams(), world_2_cam, 0.5).template head<2>().eval();
            },
            p, &J_point_2);
    }

    {
        res2 = EvaluateNumeric(
            [=](auto aff) { return ProjectOCam<double>(p, aff, world_2_cam, 0.5).template head<2>().eval(); },
            cam.AffineParams(), &J_affine_2);
    }


    ExpectCloseRelative(res1, res2, 1e-5);
    ExpectCloseRelative(J_point_1, J_point_2, 1e-5);
    ExpectCloseRelative(J_affine_1, J_affine_2, 1e-5);
}


}  // namespace Saiga
