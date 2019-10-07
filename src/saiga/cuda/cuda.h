/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

//#include <cuda.h>
#include <cuda_runtime.h>

#define SAIGA_CUDA_INCLUDED


#if !defined(__host__)
#    define __host__
#endif
#if !defined(__device__)
#    define __device__
#endif
#if !defined(__launch_bounds__)
#    define __launch_bounds__
#endif
