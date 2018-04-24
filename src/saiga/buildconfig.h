/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#define SAIGA_VERSION_MAJOR 1
#define SAIGA_VERSION_MINOR 1

//############# Libraries found ###############

//opengl window and context managment
#define SAIGA_USE_SDL
/* #undef SAIGA_USE_GLFW */

//opengl loader
/* #undef SAIGA_USE_GLBINDING */
#define SAIGA_USE_GLEW

//sound
#define SAIGA_USE_OPENAL
#define SAIGA_USE_ALUT
#define SAIGA_USE_OPUS

//asset loading
#define SAIGA_USE_ASSIMP
#define SAIGA_USE_OPENMESH

//image loading
#define SAIGA_USE_PNG
#define SAIGA_USE_FREEIMAGE

#define SAIGA_USE_FFMPEG

#define SAIGA_USE_CUDA
#define SAIGA_USE_EIGEN

//############# Build Options ###############

#define SAIGA_BUILD_SHARED
/* #undef SAIGA_DEBUG */
#define SAIGA_ASSERTS
#define SAIGA_BUILD_SAMPLES
#define SAIGA_WITH_CUDA
/* #undef SAIGA_STRICT_FP */
/* #undef SAIGA_FULL_OPTIMIZE */
#define SAIGA_CUDA_DEBUG

