/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#include "saiga/core/Core.h"
#include "saiga/core/math/all.h"
#include "saiga/normal_packing.h"

#include "gtest/gtest.h"

using namespace Saiga;


template <typename T, typename G>
void CheckNormalPacking(T pack, G unpack, int its)
{
    Random::setSeed(9346609476);
    std::cout << std::endl;
    Table tab({14, 14, 14, 4, 14, 14, 4, 14, 14, 14, 4, 14});

    tab << "Input"
        << ""
        << ""
        << ""
        << "Encoding"
        << ""
        << ""
        << "Output"
        << ""
        << ""
        << ""
        << "Error";

    std::vector<vec3> extreme_normals = {vec3(1, 0, 0),  vec3(-1, 0, 0), vec3(0, 1, 0),
                                         vec3(0, -1, 0), vec3(0, 0, 1),  vec3(0, 0, -1)};

    float max_error = 0;

    for (int i = 0; i < its; ++i)
    {
        vec3 n = Random::sphericalRand(1).cast<float>();

        if (i < extreme_normals.size())
        {
            n = extreme_normals[i];
        }

        EXPECT_NEAR(n.norm(), 1, 1e-5);

        vec2 enc = pack(n);

        if (!enc.array().allFinite())
        {
            std::cout << "non finite encoding " << enc.transpose() << std::endl;
        }
        vec3 n2 = unpack(enc);


        float diff = (n - n2).norm();

        if (diff > max_error)
        {
            max_error = diff;
        }

        EXPECT_NEAR(diff, 0, 1e-1);


        if (!std::isfinite(diff) || diff > 1e-4 || i < 10 + extreme_normals.size())
        {
            tab << n(0) << n(1) << n(2) << " -> " << enc(0) << enc(1) << " -> " << n2(0) << n2(1) << n2(2) << " = "
                << diff;
        }
        // std::cout << n.transpose() << " | " << n2.transpose() << " | " << diff << std::endl;
    }
    std::cout << "max error: " << max_error << std::endl;
    std::cout << std::endl;
}

const int its = 1000 * 1000;

TEST(NormalPacking, Spheremap)
{
    CheckNormalPacking(PackNormalSpheremap, UnpackNormalSpheremap, its);
}


TEST(NormalPacking, Stereographic)
{
    CheckNormalPacking(PackNormalStereographic, UnpackNormalStereographic, its);
}


TEST(NormalPacking, Spherical)
{
    CheckNormalPacking(PackNormalSpherical, UnpackNormalSpherical, its);
}