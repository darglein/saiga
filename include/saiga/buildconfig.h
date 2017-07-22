/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#define SAIGA_VERSION_MAJOR 1
#define SAIGA_VERSION_MINOR 0

//opengl window and context managment
#define SAIGA_USE_SDL
#define SAIGA_USE_GLFW

//opengl loader
#define SAIGA_USE_GLBINDING
/* #undef SAIGA_USE_GLEW */

//sound
#define SAIGA_USE_OPENAL
/* #undef SAIGA_USE_ALUT */
#define SAIGA_USE_OPUS

//asset loading
#define SAIGA_USE_ASSIMP

//image loading
/* #undef SAIGA_USE_PNG */
#define SAIGA_USE_FREEIMAGE

#define SAIGA_USE_FFMPEG
/* #undef SAIGA_USE_NOISE */

/* #undef SAIGA_USE_CUDA */
/* #undef SAIGA_USE_EIGEN */

/* #undef SAIGA_DEBUG */
#define SAIGA_TESTING
/* #undef SAIGA_RELEASE */

#define BUILD_SHARED
