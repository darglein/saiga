
#ifndef GLBINDING_API_H
#define GLBINDING_API_H

// Modified the orignal file here to use the saiga export macros
#include "saiga/export.h"
#define GLBINDING_API SAIGA_OPENGL_API
#define GLBINDING_NO_EXPORT SAIGA_LOCAL

#ifndef GLBINDING_DEPRECATED
#    define GLBINDING_DEPRECATED __attribute__((__deprecated__))
#endif

#ifndef GLBINDING_DEPRECATED_EXPORT
#    define GLBINDING_DEPRECATED_EXPORT GLBINDING_API GLBINDING_DEPRECATED
#endif

#ifndef GLBINDING_DEPRECATED_NO_EXPORT
#    define GLBINDING_DEPRECATED_NO_EXPORT GLBINDING_NO_EXPORT GLBINDING_DEPRECATED
#endif

#if 0 /* DEFINE_NO_DEPRECATED */
#    ifndef GLBINDING_NO_DEPRECATED
#        define GLBINDING_NO_DEPRECATED
#    endif
#endif

#endif
