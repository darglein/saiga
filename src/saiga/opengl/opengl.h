/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <saiga/config.h>

#ifndef SAIGA_USE_OPENGL
#error Saiga was build without opengl.
#endif

#include <vector>


#include <glbinding/gl/gl.h>
#include <glbinding/ProcAddress.h>
//make sure nobody else includes gl.h after this
#define __gl_h_
using namespace gl;
#define GLFW_INCLUDE_NONE



namespace Saiga {
SAIGA_GLOBAL std::ostream& operator<<(std::ostream& os, GLenum g);

SAIGA_GLOBAL void initOpenGL(glbinding::GetProcAddress func);
SAIGA_GLOBAL void terminateOpenGL();
SAIGA_GLOBAL bool OpenGLisInitialized();

SAIGA_GLOBAL int getVersionMajor();
SAIGA_GLOBAL int getVersionMinor();
SAIGA_GLOBAL void printOpenGLVersion();

SAIGA_GLOBAL int getExtensionCount();
SAIGA_GLOBAL bool hasExtension(const std::string &ext);
SAIGA_GLOBAL std::vector<std::string> getExtensions();


//called from initSaiga
SAIGA_LOCAL void initSaigaGL(const std::string& shaderDir, const std::vector<std::string>& textureDir);
SAIGA_LOCAL void cleanupSaigaGL();



enum class OpenGLVendor{
    Nvidia,
    Ati,
    Intel,
    Mesa,
    Unknown
};

SAIGA_GLOBAL OpenGLVendor getOpenGLVendor();

struct SAIGA_GLOBAL OpenGLParameters
{
    enum class Profile{
        ANY,
        CORE,
        COMPATIBILITY
    };
    Profile profile = Profile::CORE;

    bool debug = true;

    //all functionality deprecated in the requested version of OpenGL is removed
    bool forwardCompatible = false;

    int versionMajor = 3;
    int versionMinor = 2;

    /**
     *  Reads all paramters from the given config file.
     *  Creates the file with the default values if it doesn't exist.
     */
    void fromConfigFile(const std::string& file);
};

}

#define SAIGA_OPENGL_INCLUDED
