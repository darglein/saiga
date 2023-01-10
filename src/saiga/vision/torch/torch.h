#pragma once
#include "saiga/core/math/math.h"

#include "torch/torch.h"
// #include "c10/util/Logging.h"


#if TORCH_VERSION_MAJOR > 1 || TORCH_VERSION_MINOR >= 13
#    ifndef CHECK_EQ
#        define CHECK_EQ TORCH_CHECK_EQ
#    endif
#    ifndef CHECK_NE
#        define CHECK_NE TORCH_CHECK_NE
#    endif
#    ifndef CHECK_GT
#        define CHECK_GT TORCH_CHECK_GT
#    endif
#    ifndef CHECK_GE
#        define CHECK_GE TORCH_CHECK_GE
#    endif
#    ifndef CHECK_LE
#        define CHECK_LE TORCH_CHECK_LE
#    endif
#    ifndef CHECK_LT
#        define CHECK_LT TORCH_CHECK_LT
#    endif
#    ifndef CHECK_NOTNULL
#        define CHECK_NOTNULL TORCH_CHECK_NOTNULL
#    endif
// #define CHECK TORCH_CHECK
#endif