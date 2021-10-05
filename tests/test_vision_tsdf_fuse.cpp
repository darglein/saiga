/**
 * Copyright (c) 2021 Darius Rückert
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
        Random::setSeed(34976346);
        srand(45727);

        TemplatedImage<float> input("bar.saigai");
        depth_image.create(input.h / 2, input.w / 2);
        DMPP::scaleDown2median(input.getImageView(), depth_image.getImageView());


        // depth_image.getImageView().set(1.0f);
        scene.K = IntrinsicsPinholed(5.3887405952849110e+02, 5.3937051275591125e+02, 3.2233507920081263e+02, 2.3691517848391885e+02, 0);
        scene.K =        scene.K.scale(0.5);
        scene.dis = Distortion();

        scene.params.maxIntegrationDistance = 5;
        scene.params.voxelSize              = 0.05;
        scene.params.truncationDistance     = 0.2;
        scene.params.post_process_mesh      = false;
        scene.params.block_count            = 10000;
        scene.params.hash_size              = 10000;

        scene.params.extract_iso = 0;

        scene.params.ground_truth_fuse = true;



        DepthProcessor2::Settings settings;
        settings.cameraParameters = StereoCamera4(scene.K, 0.1 * scene.K.fx).cast<float>();
        DepthProcessor2 dp(settings);
        dp.Process(depth_image.getImageView());



        // scene.params.truncationDistance     = 1;
        FusionImage fi;
        fi.depthMap = depth_image.getImageView();
        fi.V        = SE3();

        for (int i = 0; i < 1; ++i)
        {
            fi.V.translation() += Vec3::Random() * 0.01;
            scene.images.push_back(fi);
        }
        scene.params.out_file = "tsdf.off";
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
    test->scene.params.point_based = false;
    test->scene.Fuse();
    //    test->scene.params.point_based = true;
    //    test->scene.Fuse();

    //    SparseTSDF t2 = *test->scene.tsdf;
}

TEST(TSDF, SmallHash)
{
    FusionScene scene2;
    scene2                  = test->scene;
    scene2.params.hash_size = 100;
    scene2.params.out_file  = "tsdf_small_hash.off";
    scene2.Fuse();
    EXPECT_EQ(test->scene.tsdf->current_blocks, scene2.tsdf->current_blocks);
}

TEST(TSDF, IncrementalFuse)
{
    FusionScene scene2;
    scene2 = test->scene;
    scene2.images.clear();
    scene2.params.out_file = "tsdf_incr.off";
    for (int i = 0; i < test->scene.images.size(); ++i)
    {
        scene2.FuseIncrement(test->scene.images[i], i == 0);
    }
    scene2.ExtractMesh();
    std::cout << *scene2.tsdf << std::endl;

    EXPECT_EQ(test->scene.tsdf->current_blocks, scene2.tsdf->current_blocks);
}



TEST(TSDF, LoadStore)
{
    SparseTSDF test2(10, 10, 10);

    EXPECT_TRUE(!(test2 == *test->scene.tsdf));
    EXPECT_EQ(test2, test2);
    EXPECT_EQ(*test->scene.tsdf, *test->scene.tsdf);

    test->scene.tsdf->Compact();
    test->scene.tsdf->Save("tsdf.dat");
    test2.Load("tsdf.dat");
    EXPECT_EQ(test2, *test->scene.tsdf);
    test2.blocks[12].data[5][3][1].distance = 10;
    EXPECT_TRUE(!(test2 == *test->scene.tsdf));


#ifdef SAIGA_USE_ZLIB
    test->scene.tsdf->SaveCompressed("tsdf_comp.dat");
    SparseTSDF tsdf3;
    tsdf3.LoadCompressed("tsdf_comp.dat");
    EXPECT_EQ(tsdf3, *test->scene.tsdf);
#endif
}


}  // namespace Saiga

int main()
{
    Saiga::initSaigaSampleNoWindow();
    testing::InitGoogleTest();

    return RUN_ALL_TESTS();
}
