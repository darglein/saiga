/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/Core.h"
#include "saiga/vision/reconstruction/MarchingCubes.h"
#include "saiga/vision/reconstruction/SparseTSDF.h"
#include "saiga/vision/reconstruction/VoxelFusion.h"

#include "gtest/gtest.h"

#include "compare_numbers.h"

namespace Saiga
{
class TSDFTest
{
   public:
    TSDFTest()
    {
        depth_image.load("bar.saigai");
        scene.K =
            Intrinsics4(5.3887405952849110e+02, 5.3937051275591125e+02, 3.2233507920081263e+02, 2.3691517848391885e+02);
        scene.dis = Distortion();

        scene.params.maxIntegrationDistance = 5;
        scene.params.voxelSize              = 0.01;
        // scene.params.truncationDistance     = 1;
        FusionImage fi;
        fi.depthMap = depth_image.getImageView();
        fi.V        = SE3();

        for (int i = 0; i < 10; ++i)
        {
            scene.images.push_back(fi);
        }
    }

    TemplatedImage<float> depth_image;
    StereoCamera4f camera;
    FusionScene scene;
};

std::unique_ptr<TSDFTest> test;

TEST(TSDF, Create)
{
    test = std::make_unique<TSDFTest>();
    EXPECT_TRUE(test->depth_image.valid());
}

TEST(TSDF, Fuse)
{
    test->scene.Fuse();
}

TEST(TSDF, LoadStore)
{
    SparseTSDF test2(10, 10, 10);

    EXPECT_TRUE(!(test2 == *test->scene.tsdf));
    EXPECT_EQ(test2, test2);
    EXPECT_EQ(*test->scene.tsdf, *test->scene.tsdf);

    test->scene.tsdf->Save("tsdf.dat");
    test2.Load("tsdf.dat");
    EXPECT_EQ(test2, *test->scene.tsdf);

    test2.blocks[12].data[5][3][1].distance = 10;
    EXPECT_TRUE(!(test2 == *test->scene.tsdf));
}


}  // namespace Saiga

int main()
{
    Saiga::initSaigaSampleNoWindow();
    testing::InitGoogleTest();

    return RUN_ALL_TESTS();
}
