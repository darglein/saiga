/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/assert.h"
#include "saiga/vision/recursiveMatrices/MatrixScalar.h"


#define SAIGA_RM_CREATE_RETURN(_LHS, _RHS, _RETURN)          \
    template <typename BinaryOp>                             \
    struct Eigen::ScalarBinaryOpTraits<_LHS, _RHS, BinaryOp> \
    {                                                        \
        typedef _RETURN ReturnType;                          \
    };
