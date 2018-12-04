/*
 * This is a debug header included by some saiga modules
 * to check if a graphics API was referenced.
 */

#if defined(SAIGA_VULKAN_INCLUDED) || defined(SAIGA_OPENGL_INCLUDED)
#    error This module must be independent of any graphics API.
#endif
