/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#if defined(_OPENMP)
#    include <omp.h>
#    define SAIGA_HAS_OMP
#endif

#include "saiga/core/util/env.h"
/**
 * This is a preprocessor wrapper for openmp.
 * With that we can make sure code runs with and without openmp.
 * This is mainly used for debugging and single-core performance analysing,
 * because we can remove openmp overhead completely. (by disabling the compiler flag)
 */
namespace Saiga
{
namespace OMP
{
inline int getMaxThreads()
{
#ifdef SAIGA_HAS_OMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}

inline int getThreadNum()
{
#ifdef SAIGA_HAS_OMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

inline int getNumThreads()
{
#ifdef SAIGA_HAS_OMP
    return omp_get_num_threads();
#else
    return 1;
#endif
}

inline void setNumThreads(int t)
{
#ifdef SAIGA_HAS_OMP
    omp_set_num_threads(t);
#else
#endif
}


enum class WaitPolicy
{
    Default,
    Active,
    Passive,
};

inline void setWaitPolicy(WaitPolicy p)
{
    switch (p)
    {
        case WaitPolicy::Default:
            SetEnv("OMP_WAIT_POLICY", "DEFAULT", true);
            break;
        case WaitPolicy::Active:
            SetEnv("OMP_WAIT_POLICY", "ACTIVE", true);
            break;
        case WaitPolicy::Passive:
            SetEnv("OMP_WAIT_POLICY", "PASSIVE", true);
            break;
    }
}



}  // namespace OMP
}  // namespace Saiga
