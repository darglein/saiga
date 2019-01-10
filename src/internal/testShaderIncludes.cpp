#include "saiga_buildconfig.h"

// Only test this if we have glm
#ifndef SAIGA_FULL_EIGEN

#ifdef _WIN32
// TODO fix for windows
#    include "saiga/shaderConfig.h"
#else
#    include "saiga/colorize.h"
#    include "saiga/hlslDefines.h"
#    include "saiga/hsv.h"
#    include "saiga/normal_sf.h"
#    include "saiga/shaderConfig.h"
#endif


/**
 * This is just a debug source file to check if all headers in saiga/shader/include/saiga
 * compile with the host compiler.
 */
#endif
