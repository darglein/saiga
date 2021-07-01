/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/image/ImageDraw.h"
#include "saiga/core/image/image.h"
#include "saiga/core/math/imath.h"
#include "saiga/core/math/math.h"
#include "saiga/core/util/assert.h"

namespace Saiga
{
// Simple Polynomial Model Based on
// Vignette and Exposure Calibration and Compensation
// https://grail.cs.washington.edu/projects/vignette/vign.iccv05.pdf
//
// r_sqared is the (squared) distance to the optical center.
// if coefficients == 0 -> No vignetting exists
using VignetteCoefficients = vec3;
inline float VignetteModel(float r_squared, VignetteCoefficients coefficients)
{
    float r2 = r_squared;
    float r4 = r2 * r2;
    float r6 = r4 * r2;
    return 1.0f + coefficients[0] * r2 + coefficients[1] * r4 + coefficients[2] * r6;
}



template <typename T = double>
struct DiscreteResponseFunction
{
    DiscreteResponseFunction(int resolution = 256)
    {
        // Linear response
        irradiance.resize(resolution);
        for (int i = 0; i < resolution; ++i)
        {
            irradiance[i] = i;
        }
    }

    T operator()(int image_intensity) { return irradiance[image_intensity]; }

    // Interpolated Read.
    // Input u must be in range [0,1]
    T NormalizedRead(float u)
    {
        u = clamp(u, 0.f, 1.f);
        SAIGA_ASSERT(u >= 0 && u <= 1);
        u = u * (irradiance.size() - 1);

        int idown = (int)u;
        int iup   = Saiga::iCeil(u);

        float alpha = u - idown;

        T a = irradiance[idown];
        T b = irradiance[iup];

        return alpha * b + (T(1) - alpha) * a;
    }

    T NormalizedLeakyRead(float u, float leak_factor)
    {
        if (u < 0)
        {
            return leak_factor * u;
        }
        else if (u > 1)
        {
            return 1 + (u - 1) * leak_factor;
        }

        return NormalizedRead(u);
    }

    DiscreteResponseFunction<T>& normalize(T target = 255)
    {
        SAIGA_ASSERT(irradiance.back() > 0);
        for (auto& d : irradiance)
        {
            d = d / irradiance.back() * target;
        }
        return *this;
    }

    DiscreteResponseFunction<T>& MakeGamma(T gamma)
    {
        SAIGA_ASSERT(irradiance.size() > 0);
        // make sure this is exact
        irradiance.front() = 0;
        irradiance.back()  = 1;
        for (int i = 1; i < irradiance.size() - 1; ++i)
        {
            float alpha   = float(i) / (irradiance.size() - 1);
            irradiance[i] = pow(alpha, gamma);
        }
        return *this;
    }

    TemplatedImage<ucvec3> Image(int n = 256, ucvec3 background_color = ucvec3(0, 0, 0),
                                 ucvec3 color = ucvec3(255, 255, 255)) const
    {
        TemplatedImage<ucvec3> img(n, n);
        img.getImageView().template set<>(background_color);
        float factor_x = float(n - 1) / (irradiance.size() - 1);
        float factor_y = float(n - 1) / (irradiance.back());

        for (int i = 0; i < irradiance.size() - 1; ++i)
        {
            vec2 start(i * factor_x, irradiance[i] * factor_y);
            vec2 end((i + 1) * factor_x, irradiance[i + 1] * factor_y);
            ImageDraw::drawLineBresenham(img.getImageView(), start, end, color);
        }
        img.getImageView().flipY();
        return img;
    }

    // Expects 3 response functions for r,g,b respectively
    static TemplatedImage<ucvec3> RGBImage(ArrayView<DiscreteResponseFunction<T>> channel_response, int n = 256)
    {
        SAIGA_ASSERT(channel_response.size() == 3);
        TemplatedImage<ucvec3> result(n, n);
        result.makeZero();

        auto R = channel_response[0].Image(n, ucvec3(0, 0, 0), ucvec3(255, 0, 0));
        auto G = channel_response[1].Image(n, ucvec3(0, 0, 0), ucvec3(0, 255, 0));
        auto B = channel_response[2].Image(n, ucvec3(0, 0, 0), ucvec3(0, 0, 255));

        for (int i : result.rowRange())
        {
            for (int j : result.colRange())
            {
                ivec3 c(0, 0, 0);
                c += R(i, j).template cast<int>();
                c += G(i, j).template cast<int>();
                c += B(i, j).template cast<int>();
                c            = c.array().min(ivec3(255, 255, 255).array());
                result(i, j) = c.cast<unsigned char>();
            }
        }
        return result;
    }


    std::vector<T> irradiance;
};



}  // namespace Saiga
