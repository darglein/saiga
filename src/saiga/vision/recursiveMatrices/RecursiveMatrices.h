/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/assert.h"
#include "saiga/vision/Eigen_Compile_Checker.h"
#include "saiga/vision/VisionIncludes.h"

// Low Level Functions
#include "saiga/vision/recursiveMatrices/Cholesky.h"
#include "saiga/vision/recursiveMatrices/Dot.h"
#include "saiga/vision/recursiveMatrices/EigenTemplateDefines.h"
#include "saiga/vision/recursiveMatrices/EigenTemplateDefines_sparse.h"
#include "saiga/vision/recursiveMatrices/Expand.h"
#include "saiga/vision/recursiveMatrices/MatrixScalar.h"
#include "saiga/vision/recursiveMatrices/MixedMatrix.h"
#include "saiga/vision/recursiveMatrices/NeutralElements.h"
#include "saiga/vision/recursiveMatrices/Norm.h"
#include "saiga/vision/recursiveMatrices/ScalarMult.h"
#include "saiga/vision/recursiveMatrices/SparseHelper.h"
#include "saiga/vision/recursiveMatrices/Transpose.h"

// High Level Functions
#include "saiga/vision/recursiveMatrices/CG.h"
#include "saiga/vision/recursiveMatrices/LM.h"
#include "saiga/vision/recursiveMatrices/MixedSolver.h"
#include "saiga/vision/recursiveMatrices/SparseCholesky.h"
