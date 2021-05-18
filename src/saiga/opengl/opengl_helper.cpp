/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "opengl_helper.h"

#include "saiga/core/util/ConsoleColor.h"
#include "saiga/core/util/assert.h"
#include "saiga/core/util/ini/ini.h"
#include "saiga/core/util/table.h"
#include "saiga/opengl/error.h"
#include "saiga/opengl/opengl.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/core/util/easylogging++.h"
#include <algorithm>
#include <glbinding/glbinding.h>


namespace gl
{
std::ostream& operator<<(std::ostream& os, GLenum g)
{
    os << (int)g;
    return os;
}
}  // namespace gl

namespace Saiga
{
bool openglinitialized = false;

void initOpenGL(glbinding::GetProcAddress func)
{
    SAIGA_ASSERT(!openglinitialized);


    glbinding::initialize(func);


    openglinitialized = true;
    printOpenGLVersion();

    switch (getOpenGLVendor())
    {
        case OpenGLVendor::Nvidia:
            Shader::add_glsl_line_directives = true;
            VLOG(1) << "Enabling #line directives for NVIDIA driver.";
            break;
        default:
            break;
    }
}

void printOpenGLVersion()
{
    std::cout << ConsoleColor::YELLOW;
    Table table({2, 18, 41, 2});
    std::cout << "=========================== OpenGL ===========================" << std::endl;
    table << "|"
          << "OpenGL Version" << glGetString(GL_VERSION) << "|";
    table << "|"
          << "GLSL Version" << glGetString(GL_SHADING_LANGUAGE_VERSION) << "|";
    table << "|"
          << "Renderer" << glGetString(GL_RENDERER) << "|";
    table << "|"
          << "Vendor" << glGetString(GL_VENDOR) << "|";
    std::cout << "==============================================================" << std::endl;
    std::cout << ConsoleColor::RESET;
}



void terminateOpenGL()
{
    SAIGA_ASSERT(openglinitialized);
    openglinitialized = false;
}

bool OpenGLisInitialized()
{
    return openglinitialized;
}

int getVersionMajor()
{
    int v;
    glGetIntegerv(GL_MAJOR_VERSION, &v);
    return v;
}

int getVersionMinor()
{
    int v;
    glGetIntegerv(GL_MINOR_VERSION, &v);
    return v;
}


int getExtensionCount()
{
    GLint n = 0;
    glGetIntegerv(GL_NUM_EXTENSIONS, &n);
    return n;
}

bool hasExtension(const std::string& ext)
{
    int n = getExtensionCount();
    for (GLint i = 0; i < n; i++)
    {
        const char* extension = (const char*)glGetStringi(GL_EXTENSIONS, i);
        if (ext == std::string(extension))
        {
            return true;
        }
    }
    return false;
}

std::vector<std::string> getExtensions()
{
    // std::ofstream myfile;
    // myfile.open ("opengl-extensions.txt");

    std::vector<std::string> extensions;


    int n = getExtensionCount();
    for (GLint i = 0; i < n; i++)
    {
        const char* extension = (const char*)glGetStringi(GL_EXTENSIONS, i);
        extensions.push_back(extension);
        // myfile << extension<<endl;
    }

    return extensions;
    // myfile.close();
}

OpenGLVendor getOpenGLVendor()
{
    std::string ven = reinterpret_cast<const char*>(glGetString(GL_VENDOR));
    std::transform(ven.begin(), ven.end(), ven.begin(), ::tolower);


    if (ven.find("nvidia") != std::string::npos)
    {
        return OpenGLVendor::Nvidia;
    }

    if (ven.find("ati") != std::string::npos)
    {
        return OpenGLVendor::Ati;
    }


    if (ven.find("intel") != std::string::npos)
    {
        return OpenGLVendor::Intel;
    }


    if (ven.find("mesa") != std::string::npos)
    {
        return OpenGLVendor::Mesa;
    }


    return OpenGLVendor::Unknown;
}


void OpenGLParameters::fromConfigFile(const std::string& file)
{
    std::string profileString;
    switch (profile)
    {
        case Profile::ANY:
            profileString = "ANY";
            break;
        case Profile::CORE:
            profileString = "saiga_core";
            break;
        case Profile::COMPATIBILITY:
            profileString = "COMPATIBILITY";
            break;
    }

    Saiga::SimpleIni ini;
    ini.LoadFile(file.c_str());

    debug             = ini.GetAddBool("opengl", "debug", debug);
    assertAtError     = ini.GetAddBool("opengl", "assertAtError", assertAtError);
    forwardCompatible = ini.GetAddBool("opengl", "forwardCompatible", forwardCompatible);
    versionMajor      = ini.GetAddLong("opengl", "versionMajor", versionMajor);
    versionMinor      = ini.GetAddLong("opengl", "versionMinor", versionMinor);
    profileString     = ini.GetAddString("opengl", "profile", profileString.c_str(),
                                     "# One of the following: 'ANY' 'CORE' 'COMPATIBILITY'");



    if (ini.changed()) ini.SaveFile(file.c_str());


    profile = profileString == "ANY"          ? Profile::ANY
              : profileString == "saiga_core" ? Profile::CORE
                                              : Profile::COMPATIBILITY;
}

void initSaigaGL(const OpenGLParameters& params)
{
    //    shaderPathes.addSearchPath(shaderDir);
    // Disables the following notification:
    // Buffer detailed info : Buffer object 2 (bound to GL_ELEMENT_ARRAY_BUFFER_ARB, usage hint is GL_STREAM_DRAW)
    // will use VIDEO memory as the source for buffer object operations.
    std::vector<GLuint> ignoreIds = {
        131185,  // nvidia

        // Vertex shader in program xx is being recompiled based on GL state.
        131218,
    };
    Error::ignoreGLError(ignoreIds);

    Error::setAssertAtError(params.assertAtError);
}

void cleanupSaigaGL()
{
    shaderLoader.clear();
    //    shaderLoader.clear();
    //    TextureLoader::instance()->clear();
}



}  // namespace Saiga
