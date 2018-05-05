/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#include <thrust/device_vector.h>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace Saiga {

//currently not working (maybe in later cuda releases)
template<typename T>
struct cr_array_view{
public:
    const T* __restrict__ data_;
    int n;
};

}
