/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <cstddef>
#include <cstdint>

#include <thrust/device_vector.h>
#include <type_traits>

namespace Saiga
{
// currently not working (maybe in later cuda releases)
template <typename T>
struct cr_ArrayView
{
   public:
    const T* __restrict__ data_;
    int n;
};

}  // namespace Saiga
