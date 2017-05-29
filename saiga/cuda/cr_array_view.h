#pragma once

#include "saiga/cuda/common.h"

#include <thrust/device_vector.h>

#include <cstddef>
#include <cstdint>
#include <type_traits>


//Thanks to Johannes Pieger

template<typename T>
struct cr_array_view{
public:
    const T* __restrict__ data_;
    int n;
};



