/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */



#include "saiga/vision/camera/KinectAzure.h"

#include "gtest/gtest.h"

#include "compare_numbers.h"
namespace Saiga
{
std::unique_ptr<KinectCamera> camera;
TEST(Azure, Open)
{
    try
    {
        KinectCamera::KinectParams params;
        camera = std::make_unique<KinectCamera>(params);
    }
    catch (std::exception e)
    {
        std::cout << "Could not open Kinect. Aborting test. " << e.what() << std::endl;
        camera = nullptr;
    }
}

TEST(Azure, ImageToWorld)
{
    if (!camera) return;

    auto calib = camera->GetCalibration();
    auto intr  = camera->intrinsics();

    for (int i = 0; i < 10; ++i)
    {
        Vec2 p       = Vec2(Random::sampleDouble(0, 500), Random::sampleDouble(0, 500));
        double depth = Random::sampleDouble(1, 5);


        k4a_float2_t px;
        px.xy.x = p.x();
        px.xy.y = p.y();
        k4a_float3_t target;
        calib.convert_2d_to_3d(px, depth * 1000, K4A_CALIBRATION_TYPE_COLOR, K4A_CALIBRATION_TYPE_COLOR, &target);
        Vec3 result(target.v[0], target.v[1], target.v[2]);
        result *= (1.0 / 1000);


        // 1. Unproject
        Vec2 norm = intr.model.K.unproject2(p);

        // 2. Undistort
        norm = undistortPointGN(norm, norm, intr.model.dis);

        // 3. Mult by depth
        Vec3 res3(norm(0), norm(1), 1);
        res3 *= depth;

        ExpectCloseRelative(result, res3, 1e-5, false);
    }
}



}  // namespace Saiga
