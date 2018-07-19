/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

#if !defined(SAIGA_USE_CUDA)
#error Saiga was compiled without cuda.
#endif

#include <cuda.h>
#include <cuda_runtime.h>

#define SAIGA_CUDA_INCLUDED
