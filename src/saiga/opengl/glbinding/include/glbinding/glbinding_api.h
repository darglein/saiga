
#ifndef GLBINDING_TEMPLATE_API_H
#define GLBINDING_TEMPLATE_API_H

#include <glbinding/glbinding_export.h>

#ifdef GLBINDING_STATIC_DEFINE
#  define GLBINDING_TEMPLATE_API
#else
#  ifndef GLBINDING_TEMPLATE_API
#    ifdef GLBINDING_EXPORTS
        /* We are building this library */
#      define GLBINDING_TEMPLATE_API __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define GLBINDING_TEMPLATE_API __attribute__((visibility("default")))
#    endif
#  endif

#endif

#endif
