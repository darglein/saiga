/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/saiga_modules.h"

// Sophus is a header only library which is not so common. Therefore we include it here, but
// only use it if cmake cannot find an installed version.
#ifdef SAIGA_SYSTEM_EIGENRECURSIVE
#    include "EigenRecursive/All.h"
#else
#    include "External/All.h"
#endif
