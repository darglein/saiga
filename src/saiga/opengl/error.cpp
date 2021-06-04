/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/error.h"

#include "saiga/core/util/tostring.h"
#include "saiga/opengl/opengl_helper.h"

#include <iostream>

#include <unordered_map>


namespace Saiga
{
static std::string getStringForSource(GLenum source)
{
    switch (source)
    {
        case GL_DEBUG_SOURCE_API_ARB:
            return ("API");
        case GL_DEBUG_SOURCE_WINDOW_SYSTEM_ARB:
            return ("Window System");
        case GL_DEBUG_SOURCE_SHADER_COMPILER_ARB:
            return ("Shader Compiler");
        case GL_DEBUG_SOURCE_THIRD_PARTY_ARB:
            return ("Third Party");
        case GL_DEBUG_SOURCE_APPLICATION_ARB:
            return ("Application");
        case GL_DEBUG_SOURCE_OTHER_ARB:
            return ("Other");
        default:
            return (to_string(int(source)));
    }
}

static std::string getStringForType(GLenum type)
{
    switch (type)
    {
        case GL_DEBUG_TYPE_ERROR_ARB:
            return ("Error");
        case GL_DEBUG_TYPE_DEPRECATED_BEHAVIOR_ARB:
            return ("Deprecated Behaviour");
        case GL_DEBUG_TYPE_UNDEFINED_BEHAVIOR_ARB:
            return ("Undefined Behaviour");
        case GL_DEBUG_TYPE_PORTABILITY_ARB:
            return ("Portability Issue");
        case GL_DEBUG_TYPE_PERFORMANCE_ARB:
            return ("Performance Issue");
        case GL_DEBUG_TYPE_OTHER_ARB:
            return "Other";
        default:
            return (to_string(int(type)));
    }
}



static std::string getStringForSeverity(GLenum severity)
{
    switch (severity)
    {
        case GL_DEBUG_SEVERITY_HIGH_ARB:
            return ("High");
        case GL_DEBUG_SEVERITY_MEDIUM_ARB:
            return ("Medium");
        case GL_DEBUG_SEVERITY_LOW_ARB:
            return ("Low");
        case GL_DEBUG_SEVERITY_NOTIFICATION:
            return "Notification";
        default:
            return (to_string(int(severity)));
    }
}



static bool assertAtError = false;
static std::unordered_map<GLuint, bool> ignoreMap;

void Error::ignoreGLError(std::vector<GLuint>& ids)
{
    for (auto i : ids)
    {
        ignoreMap[i] = true;
    }
}

void Error::setAssertAtError(bool v)
{
    assertAtError = v;
}

void Error::DebugLogConst(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message,
                          const GLvoid* userParam)
{
    (void)userParam;
    (void)length;

    if (severity == GL_DEBUG_SEVERITY_NOTIFICATION) return;
    if (ignoreMap.find(id) != ignoreMap.end()) return;



    auto typestr = getStringForType(type);
    std::cout << "GL ERROR : "
              << "[ID,type,source,severity] [" << id << "," << typestr << "," << getStringForSource(source) << ","
              << getStringForSeverity(severity) << "]" << std::endl;


    std::cout << "Message  : [" << message << "]" << std::endl;

    if (assertAtError)
    {
        SAIGA_ASSERT(0, "GL Error");
    }
}

void Error::DebugLog(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message,
                     GLvoid* userParam)
{
    (void)length;  // unused variables
    (void)userParam;

    DebugLogConst(source, type, id, severity, length, message, userParam);
}

bool Error::checkGLError()
{
    // don't call glGetError when OpenGL is not initialized
    if (!OpenGLisInitialized())
    {
        return false;
    }

    GLenum errCode;
    if ((errCode = glGetError()) != GL_NO_ERROR)
    {
        std::cout << "OpenGL error: " << errCode << std::endl;
        return true;
    }
    return false;
}


}  // namespace Saiga
