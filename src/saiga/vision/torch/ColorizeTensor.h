/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/colorize.h"

#include "ImageTensor.h"

namespace Saiga
{

// Applies the colorize function to every element.
// Adds a new dimension of size 3 to the front
template <typename ColorizeFunc>
inline torch::Tensor ColorizeTensor(torch::Tensor input, ColorizeFunc func = colorizeTurbo)
{
    auto t = input.cpu().reshape({-1}).to(torch::kFloat);

    int n    = t.size(0);
    auto out = torch::empty({3, n});

    float* in_ptr  = t.data_ptr<float>();
    float* out_ptr = out.data_ptr<float>();

    for (int i = 0; i < n; ++i)
    {
        float f = in_ptr[i];
        vec3 c  = func(f);
        for (int k = 0; k < 3; ++k)
        {
            out_ptr[k * n + i] = c(k);
        }
    }


    auto in_shape = input.sizes().vec();
    in_shape.insert(in_shape.begin(), 3);
    return out.reshape(in_shape);
}

template <typename ColorizeFunc>
inline TemplatedImage<ucvec3> ColorizeImage(ImageView<float> input, ColorizeFunc func = colorizeTurbo)
{
    TemplatedImage<ucvec3> result(input.dimensions());

    for (int i : input.rowRange())
    {
        for (int j : input.colRange())
        {
            float f      = input(i, j);
            f            = clamp(f, 0, 1);
            vec3 c       = func(f);
            c            = c.array().min((vec3(1, 1, 1))).max(vec3(0, 0, 0));
            result(i, j) = (c * 255).cast<unsigned char>();
        }
    }
    return result;
}

}  // namespace Saiga
