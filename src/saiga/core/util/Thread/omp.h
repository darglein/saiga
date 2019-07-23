/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#if defined(_OPENMP)
#    include <omp.h>
#    define SAIGA_HAS_OMP
#endif

/**
 * This is a preprocessor wrapper for openmp.
 * With that we can make sure code runs with and without openmp.
 * This is mainly used for debugging and single-core performance analysing,
 * because we can remove openmp overhead completely. (by disabling the compiler flag)
 */
namespace Saiga
{
namespace OMP {

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
}
}  // namespace Saiga
