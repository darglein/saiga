/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#include "saiga/core/image/all.h"
#include "saiga/core/math/all.h"
#include "saiga/core/util/statistics.h"
#include "saiga/cuda/imageProcessing/Fast.h"
#include "saiga/cuda/imageProcessing/ORBExtractorGPU.h"
#include "saiga/cuda/imageProcessing/OrbDescriptors.h"
#include "saiga/cuda/imageProcessing/image.h"
#include "saiga/cuda/thrust_helper.h"
#include "saiga/vision/features/OrbDescriptors.h"

#include "gtest/gtest.h"


using namespace Saiga;

class FeatureTest
{
   public:
    FeatureTest()
    {
        // Load images
        random.create(200, 200);
        for (int i : random.rowRange())
        {
            for (int j : random.colRange())
            {
                random(i, j) = Random::uniformInt(0, 255);
            }
        }

        square.create(200, 200);
        square.makeZero();
        ImageDraw::drawCircle(square.getImageView(), vec2(100, 100), 50, 255);


        random.save("test_random.png");
        square.save("test_square.png");


        d_random.upload(random);
        d_square.upload(square);

        obj_random = d_random.GetTextureObject(cudaAddressModeClamp);
        obj_square = d_square.GetTextureObject(cudaAddressModeClamp);
    }

    TemplatedImage<unsigned char> random;
    TemplatedImage<unsigned char> square;


    CUDA::CudaImage<unsigned char> d_random;
    CUDA::CudaImage<unsigned char> d_square;

    cudaTextureObject_t obj_random;
    cudaTextureObject_t obj_square;


    ORB orb;
    Saiga::CUDA::ORB gpu_orb;
};

FeatureTest test;

TEST(ORB, Angle)
{
    vec2 center(100, 100);
    float radius = 50;

    int n = 100;

    std::vector<std::pair<vec2, float>> point_expected_angle;

    for (int i = 0; i < 100; ++i)
    {
        float alpha = (float(i) / n) * pi<float>() * 2;
        vec2 p;
        p.x() = sin(-alpha - pi<float>() / 2) * radius;
        p.y() = cos(-alpha - pi<float>() / 2) * radius;
        p += center;
        point_expected_angle.push_back({p, degrees(alpha)});
    }


    for (auto pa : point_expected_angle)
    {
        float angle = test.orb.ComputeAngle(test.square, pa.first);
        // We allow a 2 degree error due to sampling artifacts
        EXPECT_NEAR(angle, pa.second, 2);
    }



    thrust::device_vector<KeyPoint<float>> d_kps(point_expected_angle.size());
    for (int i = 0; i < point_expected_angle.size(); ++i)
    {
        KeyPoint<float> kp;
        kp.point = point_expected_angle[i].first;
        kp.angle = 0;
        d_kps[i] = kp;
    }

    test.gpu_orb.ComputeAngles(test.obj_square, test.d_square.getImageView(), d_kps, 0, 0, 0, 0, 0);


    for (int i = 0; i < point_expected_angle.size(); ++i)
    {
        KeyPoint<float> h_kp = d_kps[i];
        EXPECT_NEAR(point_expected_angle[i].second, h_kp.angle, 2);
    }
}

TEST(ORB, AngleRandom)
{
    std::vector<vec2> sample_points;

    // use enough points to fill multiple SMs on the gpu
    for (int i = 0; i < 2000; ++i)
    {
        sample_points.push_back(vec2(Random::uniformInt(50, 150), Random::uniformInt(50, 150)));
    }


    std::vector<double> angle_cpu;
    for (auto p : sample_points)
    {
        float angle = test.orb.ComputeAngle(test.random, p);
        angle_cpu.push_back(angle);
    }



    thrust::device_vector<KeyPoint<float>> d_kps(sample_points.size());
    for (int i = 0; i < sample_points.size(); ++i)
    {
        KeyPoint<float> kp;
        kp.point = sample_points[i];
        kp.angle = 0;
        d_kps[i] = kp;
    }

    test.gpu_orb.ComputeAngles(test.obj_random, test.d_random.getImageView(), d_kps, 0, 0, 0, 0, 0);


    for (int i = 0; i < sample_points.size(); ++i)
    {
        KeyPoint<float> h_kp = d_kps[i];
        EXPECT_NEAR(angle_cpu[i], h_kp.angle, 0.1);
    }
}

TEST(ORB, DescriptorRandom)
{
    thrust::host_vector<KeyPoint<float>> sample_keypoints;

    int N = 20000;

    for (int i = 0; i < N; ++i)
    {
        KeyPoint<float> kp;
        kp.point = vec2(Random::uniformInt(50, 150), Random::uniformInt(50, 150));

        if (i < N / 2)
        {
            kp.angle = 0;
        }
        else
        {
            kp.angle = Random::sampleDouble(0, 360);
        }

        sample_keypoints.push_back(kp);
    }


    std::vector<DescriptorORB> cpu_descriptors;


    for (int i = 0; i < sample_keypoints.size(); ++i)
    {
        auto kp = sample_keypoints[i];
        cpu_descriptors.push_back(test.orb.ComputeDescriptor(test.random, kp.point, kp.angle));
    }


    thrust::device_vector<KeyPoint<float>> gpu_keypoints = sample_keypoints;
    thrust::device_vector<DescriptorORB> gpu_descriptors(sample_keypoints.size());

    test.gpu_orb.ComputeDescriptors(test.obj_random, test.d_random.getImageView(), gpu_keypoints, gpu_descriptors, 0);

    thrust::host_vector<DescriptorORB> gpu_descriptors_on_host(gpu_descriptors);

    std::vector<double> distance_0;
    std::vector<double> distance_random;

    for (int i = 0; i < sample_keypoints.size(); ++i)
    {
        auto desc1 = cpu_descriptors[i];
        auto desc2 = gpu_descriptors_on_host[i];
        auto dis   = distance(desc1, desc2);
        if (i < N / 2)
        {
            distance_0.push_back(dis);
        }
        else
        {
            distance_random.push_back(dis);
        }
    }
    Statistics<double> stat_0(distance_0);
    Statistics<double> stat_random(distance_random);

    // The 0 angles must be exact 0
    EXPECT_EQ(stat_0.max, 0);

    // Random angles is a little above 0 because of sin/cos differences
    EXPECT_LE(stat_random.max, 10);
    EXPECT_EQ(stat_random.median, 0);
}

// Regression test: Fast::Download returned the number of detected keypoints even if it was larger than
// maxKeypoints. It then copied (and later scheduled copies of) more keypoints than the buffers hold.
TEST(Fast, MoreKeypointsThanMax)
{
    CUDA::CudaStream stream;
    for (int max_keypoints : {10, 100000})
    {
        CUDA::Fast fast(20, 7, max_keypoints);
        std::vector<KeyPoint<float>> keypoints(fast.MaxKeypoints());
        int last_n = -1;
        for (int i = 0; i < 3; ++i)
        {
            fast.Detect(test.d_random.getImageView(), stream);
            int n = fast.Download(keypoints, stream);
            EXPECT_GT(n, 0);
            EXPECT_LE(n, fast.MaxKeypoints());
            if (last_n >= 0) EXPECT_EQ(n, last_n);
            last_n = n;

            for (int k = 0; k < n; ++k)
            {
                auto p = keypoints[k].point;
                EXPECT_GE(p.x(), 4);
                EXPECT_GE(p.y(), 4);
                EXPECT_LT(p.x(), test.random.w - 4);
                EXPECT_LT(p.y(), test.random.h - 4);
            }
        }
    }
}

// The extractor owns texture objects of its pyramid levels, which were never destroyed. This test runs the complete
// extractor (and its destructor) several times.
TEST(ORBExtractorGPU, Detect)
{
    TemplatedImage<unsigned char> image(240, 320);
    for (int i : image.rowRange())
    {
        for (int j : image.colRange())
        {
            image(i, j) = Random::uniformInt(0, 255);
        }
    }

    for (int run = 0; run < 3; ++run)
    {
        ORBExtractorGPU extractor(500, 1.2, 4, 20, 7);
        for (int frame = 0; frame < 2; ++frame)
        {
            std::vector<KeyPoint<float>> keypoints;
            std::vector<DescriptorORB> descriptors;
            int n = extractor.Detect(image.getImageView(), keypoints, descriptors);
            EXPECT_GT(n, 100);
            EXPECT_EQ(keypoints.size(), n);
            EXPECT_EQ(descriptors.size(), n);
            // The FAST keypoints are collected with atomics, so the exact result differs slightly between runs.
            EXPECT_LE(n, 600);
            for (auto& kp : keypoints)
            {
                EXPECT_GE(kp.point.x(), 0);
                EXPECT_GE(kp.point.y(), 0);
                EXPECT_LT(kp.point.x(), image.w);
                EXPECT_LT(kp.point.y(), image.h);
            }
        }
    }
}
