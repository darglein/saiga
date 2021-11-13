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
    auto t = input.cpu().reshape({-1}).to(torch::kFloat).clamp(0, 1);

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

}  // namespace Saiga
