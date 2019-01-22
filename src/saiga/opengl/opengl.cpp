/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/opengl.h"

#include "saiga/opengl/error.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/shader/shaderPartLoader.h"
#include "saiga/opengl/texture/textureLoader.h"
#include "saiga/util/assert.h"
#include "saiga/util/ini/ini.h"

#include <algorithm>
#include <glbinding/glbinding.h>

#include <glbinding-aux/Meta.h>



namespace Saiga
{
std::ostream& operator<<(std::ostream& os, GLenum g)
{
    os << glbinding::aux::Meta::getString(g);
    return os;
}

bool openglinitialized = false;

void initOpenGL(glbinding::GetProcAddress func)
{
    SAIGA_ASSERT(!openglinitialized);


    glbinding::initialize(func);


    openglinitialized = true;
    std::cout << "> OpenGL initialized" << std::endl;
    printOpenGLVersion();

    switch (getOpenGLVendor())
    {
        case OpenGLVendor::Nvidia:
            ShaderPartLoader::addLineDirectives = true;
            std::cout << "Enabling #line directives for NVIDIA driver." << std::endl;
            break;
        default:
            break;
    }
}

void printOpenGLVersion()
{
    std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
    std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;
    std::cout << "Vendor: " << glGetString(GL_VENDOR) << " - ";

    switch (getOpenGLVendor())
    {
        case OpenGLVendor::Nvidia:
            std::cout << "Nvidia";
            break;
        case OpenGLVendor::Ati:
            std::cout << "Ati";
            break;
        case OpenGLVendor::Intel:
            std::cout << "Intel";
            break;
        case OpenGLVendor::Mesa:
            std::cout << "Mesa";
            break;
        default:
            std::cout << "Unknown";
            break;
    }
    std::cout << std::endl;
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
            profileString = "CORE";
            break;
        case Profile::COMPATIBILITY:
            profileString = "COMPATIBILITY";
            break;
    }

    Saiga::SimpleIni ini;
    ini.LoadFile(file.c_str());

    debug             = ini.GetAddBool("opengl", "debug", debug);
    forwardCompatible = ini.GetAddBool("opengl", "forwardCompatible", forwardCompatible);
    versionMajor      = ini.GetAddLong("opengl", "versionMajor", versionMajor);
    versionMinor      = ini.GetAddLong("opengl", "versionMinor", versionMinor);
    profileString     = ini.GetAddString("opengl", "profile", profileString.c_str(),
                                     "# One of the following: 'ANY' 'CORE' 'COMPATIBILITY'");

    if (ini.changed()) ini.SaveFile(file.c_str());


    profile = profileString == "ANY" ? Profile::ANY : profileString == "CORE" ? Profile::CORE : Profile::COMPATIBILITY;
}

void initSaigaGL(const std::string& shaderDir, const std::vector<std::string>& textureDir)
{
    //    shaderPathes.addSearchPath(shaderDir);
    // Disables the following notification:
    // Buffer detailed info : Buffer object 2 (bound to GL_ELEMENT_ARRAY_BUFFER_ARB, usage hint is GL_STREAM_DRAW)
    // will use VIDEO memory as the source for buffer object operations.
    std::vector<GLuint> ignoreIds = {
        131185,  // nvidia
        // intel
    };
    Error::ignoreGLError(ignoreIds);

    //    TextureLoader::instance()->addPath(".");
    //    for (auto t : textureDir) TextureLoader::instance()->addPath(t);
    //    TextureLoader::instance()->addPath(textureDir);
    //    TextureLoader::instance()->addPath(textureDir + "/include");
}

void cleanupSaigaGL()
{
    //    ShaderLoader::instance()->clear();
    //    TextureLoader::instance()->clear();
}



}  // namespace Saiga
