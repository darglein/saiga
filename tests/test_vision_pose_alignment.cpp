/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/vision/icp/ICPAlign.h"
#include "saiga/vision/slam/Trajectory.h"
#include "saiga/vision/util/Random.h"

#include "gtest/gtest.h"

#include "compare_numbers.h"

namespace Saiga
{
namespace Trajectory
{
Scene Create(int N)
{
    Scene scene;

    scene.transformation = Random::randomSE3();
    scene.scale          = Random::sampleDouble(0.2, 10);
    scene.extrinsics     = Random::randomSE3();

    for (int i = 0; i < N; ++i)
    {
        Observation obs;
        obs.estimate     = Random::randomSE3();
        obs.ground_truth = scene.TransformVertex(obs.estimate);
        scene.vertices.push_back(obs);
    }


    std::cout << scene << std::endl;
    scene.scale          = 1;
    scene.transformation = SE3();
    scene.extrinsics     = SE3();
    return scene;
}


TEST(PoseAlignment, Default)
{
    auto scene                = Create(100);
    scene.optimize_extrinsics = true;
    std::cout << scene << std::endl;
    scene.InitialAlignment();
    std::cout << scene << std::endl;
    scene.OptimizeCeres();
    std::cout << scene << std::endl;
}

}  // namespace Trajectory
}  // namespace Saiga
