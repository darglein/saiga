/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#define EIGEN_WORLD_VERSION 1
#define EIGEN_MAJOR_VERSION 0
#define EIGEN_MINOR_VERSION 0

#define TINY_EIGEN 1


#if defined(__CUDACC__)
#    if !defined(HD)
#        define HD __host__ __device__
#    endif
#else
#    if !defined(HD)
#        define HD
#    endif
#endif
