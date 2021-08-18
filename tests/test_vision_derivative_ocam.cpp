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
TEST(Derivative, Model)
{
    Random::setSeed(903476346);
    srand(976157);

    OCam<double> cam;

    Vec5 affine;
    affine << 1.0001200000e+00, 3.1232900000e-03, -3.1106100000e-03, 1.8263000000e+03, 2.7250100000e+03;
    cam.SetAffineParams(affine);

    std::vector<double> world_2_cam = {2.1853300000e+03,  1.3746500000e+03, -1.9568400000e+02, -2.7793500000e+02,
                                       1.4024400000e+02,  4.8937200000e+02, -1.0358400000e+02, -6.6795000000e+02,
                                       5.6145100000e+01,  7.5621400000e+02, 1.0581900000e+02,  -5.7444200000e+02,
                                       -2.1457300000e+02, 2.4003900000e+02, 1.5284200000e+02,  -2.7509800000e+01,
                                       -3.9366600000e+01, -7.9584200000e+00};

    std::vector<double> cam_2_world = {-1.3760700000e+03, 0.0000000000e+00,  3.5451700000e-04, -2.6874700000e-07,
                                       2.5379600000e-10,  -1.0337200000e-13, 1.6999900000e-17};

    for (int i = 0; i < 1000; ++i)
    {
        Vec2 ip  = Random::MatrixUniform<Vec2>(1000, 3000);
        double z = 1.2;

        Vec3 wp = UnprojectOCam<double>(ip, z, cam.AffineParams(), cam_2_world);


        Vec3 ip_z = ProjectOCam<double>(wp, cam.AffineParams(), world_2_cam, 0.5);
        Vec2 ip2  = ip_z.head<2>();
        double z2 = ip_z(2);

        ExpectClose(z, z2, 1e-3);

        // we allow a 0.5 pixel error
        ExpectCloseRelative(ip, ip2, 0.5, false);
    }
}
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
