/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/vision/VisionTypes.h"

#include "ceres/solver.h"

namespace Saiga {


inline void makeGaussNewtonOptions(ceres::Solver::Options &options)
{

            options.min_trust_region_radius = 1e-32;
            options.max_trust_region_radius = 1e51;

            options.initial_trust_region_radius = 1e30;








//            options.min_trust_region_radius = 10e50;
            options.min_lm_diagonal = 1e-50;
            options.max_lm_diagonal = 1e-49;
}


}
