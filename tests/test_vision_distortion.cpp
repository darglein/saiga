/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#include "saiga/core/image/all.h"
#include "saiga/vision/cameraModel/Distortion.h"
#include "saiga/vision/util/Random.h"

#include "gtest/gtest.h"

#include "compare_numbers.h"
#include "numeric_derivative.h"
using namespace Saiga;


TEST(Distortion, Derivative)
{
    Vector<double, 8> c;
    c << -0.283408, 0.5, -1, 0.2, 0.4, 0.7, 1, 2;
    //    c.setZero();
    Distortion d(c);


    Vec2 ref = Vec2::Random();
    Vec2 p   = Vec2::Random();
    Matrix<double, 2, 2> J1, J2;

    Vec2 res1 = distortNormalizedPoint2(p, d, &J1);
    Vec2 res2 = EvaluateNumeric([&](auto p) { return distortNormalizedPoint2(p, d); }, p, &J2, 1e-8);

    ExpectCloseRelative(res1, res2, 1e-5);
    ExpectCloseRelative(J1, J2, 1e-5);
}

TEST(Distortion, Solve)
{
    //    Intrinsics4 K(458.654, 457.296, 367.215, 248.375);
    Intrinsics4 K(608.894, 608.742, 638.974, 364.492);

    Vector<double, 8> c;
    //    c << -0.283408, 0.739591, -0.31, 0, 0, 0, 0.00019359, 1.76187e-05;
    c << -0.0284351, -2.47131, 1.7389, -0.145427, -2.26192, 1.63544, 0.00128128, -0.000454588;
    //    c.setRandom();
    //    c *= 0.1;

    Distortion d(c);


    Vec2 ref = Vec2::Random();
    Vec2 p   = distortNormalizedPoint(ref, d);

    Vec2 p1 = undistortNormalizedPoint(p, d, 30);
    Vec2 p2 = undistortPointGN(p, p, d, 30);


    std::cout << ref.transpose() << " | " << p.transpose() << std::endl;
    std::cout << p1.transpose() << " | " << p2.transpose() << std::endl;
    //        ExpectCloseRelative(ref, p1, 1e-200);
    //    ExpectCloseRelative(ref, p2, 1e-200);

    //    for (int i = 0; i < 100; ++i)
    for (int x = 0; x < 1280; ++x)
    {
        for (int y = 0; y < 720; ++y)
        {
            //                int x = Random::uniformInt(0, 752);
            //                int y = Random::uniformInt(0, 480);

            Vec2 p_norm = K.unproject2(Vec2(x, y));


            Vec2 p1 = undistortNormalizedPoint(p_norm, d, 10);
            Vec2 p2 = undistortPointGN(p_norm, p_norm, d, 10);

            double e1 = (distortNormalizedPoint2(p1, d) - p_norm).norm();
            double e2 = (distortNormalizedPoint2(p2, d) - p_norm).norm();


            std::cout << e2 << std::endl;
            //        std::cout << p1.transpose() << " " << p2.transpose() << " " << e1 << " " << e2 << std::endl;
            //        EXPECT_LE(e1, 1e-3);
            EXPECT_LE(e2, 1e-3);
        }
    }
    exit(0);
}


TEST(Distortion, Brown5)
{
    // EuRoC camera parameters (+k3)
    Intrinsics4 K(458.654, 457.296, 367.215, 248.375);


    Vector<double, 8> c;
    c << -0.283408, 0.0739591, -0.031, 0, 0, 0, 0.00019359, 1.76187e-05;
    Distortion d(c);


    ExpectCloseRelative(c, d.Coeffs(), 1e-20);

    int w = 752;
    int h = 480;


    TemplatedImage<float> undistort_map_x(h, w);
    TemplatedImage<float> undistort_map_y(h, w);

    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            undistort_map_y(i, j) = i;
            undistort_map_x(i, j) = j;
        }
    }

#if 1
    int s = 0;

    // Build undistort map
    for (int i = -(s * h); i < ((s + 1) * h); ++i)
    {
        for (int j = -(s * w); j < ((s + 1) * w); ++j)
        {
            Vec2 p     = K.unproject2(Vec2(j, i));
            Vec2 p_dis = distortNormalizedPoint(p, d);
            Vec2 ip    = K.normalizedToImage(p_dis);

            int x = iRound(ip.x());
            int y = iRound(ip.y());

            if (undistort_map_x.inImage(y, x))
            {
                undistort_map_y(y, x) = i;
                undistort_map_x(y, x) = j;
            }
        }
    }
#endif


    auto optimize = [=](const Vec2& p, const Vec2& g) {
        return K.normalizedToImage(undistortPointGN(K.unproject2(p), K.unproject2(g), d, 5));
    };
    auto error = [=](const Vec2& p, const Vec2& p_un) {
        return (K.normalizedToImage(distortNormalizedPoint(K.unproject2(p_un), d)) - p).squaredNorm();
    };

    for (int k = 0; k < 10; ++k)
    {
        // Propagate forward
        for (int i = 1; i < h; ++i)
        {
            for (int j = 1; j < w; ++j)
            {
                Vec2 p(j, i);
                Vec2 map_c = Vec2(undistort_map_x(i, j), undistort_map_y(i, j));
                Vec2 map_l = Vec2(undistort_map_x(i - 1, j), undistort_map_y(i - 1, j));
                Vec2 map_u = Vec2(undistort_map_x(i, j - 1), undistort_map_y(i, j - 1));


                //                map_l += Vec2::Random() * 3;
                //                map_u += Vec2::Random() * 3;

                map_c = optimize(p, map_c);
                map_l = optimize(p, map_l);
                map_u = optimize(p, map_u);

                auto e1 = error(p, map_c);
                auto e2 = error(p, map_l);
                auto e3 = error(p, map_u);

                if (j == 645 && i == 479)
                {
                    std::cout << "forwrad " << e1 << " " << e2 << "  " << e3 << std::endl;
                }


                if (e2 < e1 && e2 <= e3)
                {
                    undistort_map_x(i, j) = map_l.x();
                    undistort_map_y(i, j) = map_l.y();
                }
                else if (e3 < e1 && e3 < e2)
                {
                    undistort_map_x(i, j) = map_u.x();
                    undistort_map_y(i, j) = map_u.y();
                }
                else
                {
                    undistort_map_x(i, j) = map_c.x();
                    undistort_map_y(i, j) = map_c.y();
                }
            }
        }


        // Propagate backward
        for (int i = h - 2; i >= 0; --i)
        {
            for (int j = h - 2; j >= 0; --j)
            {
                Vec2 p(j, i);
                Vec2 map_c = Vec2(undistort_map_x(i, j), undistort_map_y(i, j));
                Vec2 map_l = Vec2(undistort_map_x(i + 1, j), undistort_map_y(i + 1, j));
                Vec2 map_u = Vec2(undistort_map_x(i, j + 1), undistort_map_y(i, j + 1));

                map_c = optimize(p, map_c);
                map_l = optimize(p, map_l);
                map_u = optimize(p, map_u);

                auto e1 = error(p, map_c);
                auto e2 = error(p, map_l);
                auto e3 = error(p, map_u);

                //            std::cout << "forward " << e1 << " " << e2 << " " << e3 << std::endl;
                if (e2 < e1 && e2 <= e3)
                {
                    undistort_map_x(i, j) = map_l.x();
                    undistort_map_y(i, j) = map_l.y();
                }
                else if (e3 < e1 && e3 < e2)
                {
                    undistort_map_x(i, j) = map_u.x();
                    undistort_map_y(i, j) = map_u.y();
                }
                else
                {
                    undistort_map_x(i, j) = map_c.x();
                    undistort_map_y(i, j) = map_c.y();
                }
            }
        }
    }

    for (int i = 0; i < h; ++i)
    {
        for (int j = 0; j < w; ++j)
        {
            Vec2 p(j, i);



            Vec2 p_norm     = K.unproject2(Vec2(j, i));
            Vec2 p_norm_map = K.unproject2(Vec2(undistort_map_x(i, j), undistort_map_y(i, j)));


            Vec2 p1 = undistortNormalizedPoint(p_norm, d, 30);
            Vec2 p2 = undistortPointGN(p_norm, p_norm, d, 30);
            Vec2 p3 = undistortPointGN(p_norm, p_norm_map, d, 30);
            Vec2 p4 = Vec2(-1.62561, -0.98104);



            double e1 = (distortNormalizedPoint(p1, d) - p_norm).norm();
            double e2 = (distortNormalizedPoint(p2, d) - p_norm).norm();
            double e3 = (distortNormalizedPoint(p3, d) - p_norm).norm();
            double e4 = (distortNormalizedPoint(p4, d) - p_norm).norm();


            //            EXPECT_LE(e3, 1e-3);
            //            EXPECT_LE(e2, 1e-3);


            //            std::cout << p_norm.transpose() << " " << p2.transpose() << " " << p3.transpose() <<
            //            std::endl;

            if (j == 645 && i == 479)
            {
                if (e3 > 1e-3)
                {
                    //                std::cout << "es " << std::endl;
                    std::cout << p.transpose() << " | " << e1 << " " << e2 << " " << e3 << " " << e4 << std::endl;
                }
            }


#if 0
            {
                Vec2 p1 = K.normalizedToImage(undistortNormalizedPoint(K.unproject2(p), d, 30));
                Vec2 p2 = K.normalizedToImage(undistortPointGN(K.unproject2(p), K.unproject2(p), d, 30));


                //            double e1 = (K.normalizedToImage(distortNormalizedPoint(K.unproject2(map_p), d)) -
                //            p).norm();
                double e1 = (K.normalizedToImage(distortNormalizedPoint(K.unproject2(p1), d)) - p).norm();
                double e2 = (K.normalizedToImage(distortNormalizedPoint(K.unproject2(p2), d)) - p).norm();

                if (e1 > 1 || e2 > 1)
                {
                    std::cout << p.transpose() << " | " << p1.transpose() << ", " << e1 << " | " << p2.transpose()
                              << ", " << e2 << std::endl;
                }
            }

#endif

            //            Vec2 p_dis = distortNormalizedPoint(p_un, d);
            //            ExpectCloseRelative(p, p_dis, 1e-5);



            //            Vec2 p2 = undistortNormalizedPoint(distortNormalizedPoint(p, d), d, 30);
            //            ExpectCloseRelative(p, p2, 1e-5, false);
        }
    }
}
