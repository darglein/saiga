/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/util/assert.h"

#include <saiga/opengl/opengl.h>

//#include <cstdlib>
#include <vector>

namespace Saiga
{
/*
 * Use assert_no_glerror for normal gl error checking. If saiga is compiled in testing or release mode all these error
 * checks are removed. In testing mode only the error checks with assert_no_glerror_end_frame are executed.
 */

#if defined(SAIGA_DEBUG)
#    define assert_no_glerror() SAIGA_ASSERT(!Error::checkGLError())
#else
#    define assert_no_glerror() (void)0
#endif

#if defined(SAIGA_DEBUG)
#    define assert_no_glerror_end_frame() SAIGA_ASSERT(!Error::checkGLError())
#else
#    define assert_no_glerror_end_frame() (void)0
#endif



class SAIGA_OPENGL_API Error
{
   public:
    static bool checkGLError();



    // ignores all gl errors in the debug output
    static void ignoreGLError(std::vector<GLuint>& ids);

    static void setAssertAtError(bool v);

    static void DebugLogConst(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length,
                              const GLchar* message, const GLvoid* userParam);

    static void DebugLog(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message,
                         GLvoid* userParam);
};

}  // namespace Saiga
