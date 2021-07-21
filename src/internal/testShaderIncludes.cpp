#include "saiga/saiga_buildconfig.h"

// Only test this if we have glm
// #ifdef NOT_SUPPORTED_RIGHT_NOW

#ifdef _WIN32
// TODO fix for windows
#    include "saiga/shaderConfig.h"
#else
#    include "saiga/shaderConfig.h"
//
#    include "saiga/colorize.h"
#    include "saiga/hlslDefines.h"
#    include "saiga/hsv.h"
//#    include "saiga/normal_sf.h"
#endif


/**
 * This is just a debug source file to check if all headers in saiga/shader/include/saiga
 * compile with the host compiler.
 */
// #endif
