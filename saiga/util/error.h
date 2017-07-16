#pragma once

#include "saiga/util/assert.h"
#include <saiga/opengl/opengl.h>

#include <cstdlib>

namespace Saiga {

/*
 * Use assert_no_glerror for normal gl error checking. If saiga is compiled in testing or release mode all these error checks are removed.
 * In testing mode only the error checks with assert_no_glerror_end_frame are executed.
 */

#if defined(SAIGA_DEBUG)
    #define assert_no_glerror() SAIGA_ASSERT(!Error::checkGLError())
#else
    #define assert_no_glerror() (void)0
#endif

#if defined(SAIGA_DEBUG) || defined(SAIGA_TESTING)
    #define assert_no_glerror_end_frame() SAIGA_ASSERT(!Error::checkGLError())
#else
    #define assert_no_glerror_end_frame() (void)0
#endif



class SAIGA_GLOBAL Error{
public:

    static bool checkGLError();

//    static void quitWhenError(const char* func);


    // aux function to translate source to std::string
    static std::string getStringForSource(GLenum source);

    // aux function to translate severity to std::string
    static std::string getStringForSeverity(GLenum severity);

    // aux function to translate type to std::string
    static std::string getStringForType(GLenum type);

    static void DebugLogConst( GLenum source , GLenum type , GLuint id , GLenum severity ,
                               GLsizei length , const GLchar * message ,const GLvoid * userParam);

    static void DebugLog( GLenum source , GLenum type , GLuint id , GLenum severity ,
                          GLsizei length , const GLchar * message , GLvoid * userParam);
};

}
