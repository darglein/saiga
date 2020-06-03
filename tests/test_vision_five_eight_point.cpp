/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/framework/framework.h"
#include "saiga/core/image/image.h"
#include "saiga/core/math/Eigen_Compile_Checker.h"
#include "saiga/core/math/random.h"
#include "saiga/core/time/all.h"
#include "saiga/core/util/BinaryFile.h"
#include "saiga/core/util/fileChecker.h"
#include "saiga/vision/VisionIncludes.h"
#include "saiga/vision/features/Features.h"
#include "saiga/vision/reconstruction/EightPoint.h"
#include "saiga/vision/reconstruction/FivePoint.h"
#include "saiga/vision/reconstruction/TwoViewReconstruction.h"
#include "saiga/vision/scene/Scene.h"
#include "saiga/vision/util/Random.h"

#include "gtest/gtest.h"

#include "compare_numbers.h"

namespace Saiga
{
using FeatureDescriptor = DescriptorORB;



class FiveEightPointTest
{
   public:
    FiveEightPointTest()
    {
        // The transformation we want to reconstruct.
        T = Random::randomSE3();

        // Compute reference E and F
        reference_E = EssentialMatrix(SE3(), T);
        reference_F = FundamentalMatrix(reference_E, K1, K2);
    }

    SE3 T;
    Mat3 reference_E, reference_F;

    Intrinsics4 K1 = Intrinsics4(535, 539, 320, 247);
    Intrinsics4 K2 = Intrinsics4(416, 451, 215, 230);

    std::vector<Vec2> points1, points2;
    std::vector<Vec2> normalized_points1, normalized_points2;
};

TEST(TwoViewReconstruction, Load) {}
}  // namespace Saiga
