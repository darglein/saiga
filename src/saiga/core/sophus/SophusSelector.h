/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/saiga_modules.h"

// Without this define, sophus adds a few asserts that check for valid input.
// We rather just compute with nans and infs.
#define SOPHUS_DISABLE_ENSURES

// Sophus is a header only library which is not so common. Therefore we include it here, but
// only use it if cmake cannot find an installed version.
#ifdef SAIGA_SYSTEM_SOPHUS
#    include "sophus/se3.hpp"
#    include "sophus/sim3.hpp"
#else
#    include "External/se3.hpp"
#    include "External/sim3.hpp"
#endif

