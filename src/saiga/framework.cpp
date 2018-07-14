/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/framework.h"

#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/texture/textureLoader.h"
#include "saiga/opengl/error.h"
#include "saiga/util/assert.h"

#include "saiga/cuda/tests/test.h"
#include "saiga/util/floatingPoint.h"
#include "saiga/util/ini/ini.h"
#include "saiga/util/directory.h"

#ifdef SAIGA_USE_VULKAN
#include "saiga/vulkan/Shader/GLSL.h"
#endif

namespace Saiga {

bool initialized = false;


bool isShaderDirectory(const std::string &dir)
{
    Directory dirbase(dir);
    Directory dirgeo(dir + "/geometry");
    return dirbase.existsFile("colored_points.glsl") && dirgeo.existsFile("deferred_mvp_texture.glsl");
}


void SaigaParameters::fromConfigFile(const std::string &file)
{
    Saiga::SimpleIni ini;
    ini.LoadFile(file.c_str());

    shaderDirectory   = ini.GetAddString ("saiga","shareDirectory",shaderDirectory.c_str());
    textureDirectory  = ini.GetAddString ("saiga","textureDirectory",textureDirectory.c_str());

    if(ini.changed()) ini.SaveFile(file.c_str());
}



static void printSaigaInfo(){
    cout << "Saiga Version " << SAIGA_VERSION_MAJOR << "." <<  SAIGA_VERSION_MINOR << endl;
    std::string libs;
#ifdef SAIGA_USE_SDL
    libs += "SDL,";
#endif
#ifdef SAIGA_USE_GLFW
    libs += "GLFW,";
#endif
#ifdef SAIGA_USE_OPENAL
    libs += "OPENAL,";
#endif
#ifdef SAIGA_USE_ALUT
    libs += "ALUT,";
#endif
#ifdef SAIGA_USE_OPUS
    libs += "OPUS,";
#endif
#ifdef SAIGA_USE_ASSIMP
    libs += "ASSIMP,";
#endif
#ifdef SAIGA_USE_PNG
    libs += "PNG,";
#endif
#ifdef SAIGA_USE_FREEIMAGE
    libs += "FREEIMAGE,";
#endif
#ifdef SAIGA_USE_FFMPEG
    libs += "FFMPEG,";
#endif
#ifdef SAIGA_USE_CUDA
    libs += "CUDA,";
#endif
#ifdef SAIGA_USE_EIGEN
    libs += "EIGEN,";
#endif
    cout << "Libs: " << libs << endl;

    std::string options;
#ifdef SAIGA_BUILD_SHARED
    options += "BUILD_SHARED,";
#endif
#ifdef SAIGA_DEBUG
    options += "DEBUG,";
#endif
#ifdef SAIGA_ASSERTS
    options += "ASSERTS,";
#endif
#ifdef SAIGA_BUILD_SAMPLES
    options += "BUILD_SAMPLES,";
#endif
#ifdef SAIGA_WITH_CUDA
    options += "WITH_CUDA,";
#endif
#ifdef SAIGA_STRICT_FP
    options += "STRICT_FP,";
#endif
#ifdef SAIGA_FULL_OPTIMIZE
    options += "FULL_OPTIMIZE,";
#endif
#ifdef SAIGA_CUDA_DEBUG
    options += "CUDA_DEBUG,";
#endif
    cout << "Build Options: " << options << endl;
}



void initSaiga(const SaigaParameters& params)
{
    if(initialized)
    {
        return;
    }

    FP::resetSSECSR();


    std::vector<std::string> searchPathes =
    {
        // First check in the local working directory
        "shader",
        // Then the given paramter from the config file
        params.shaderDirectory,
        // And last the install prefix from cmake
        SAIGA_INSTALL_PREFIX  "/share/saiga/shader",
    };


    std::string shaderDir;

    for(auto str : searchPathes)
    {
        if(isShaderDirectory(str))
        {
            shaderDir = str;
            cout << "Found the Saiga shaders at " << shaderDir << endl;
            break;
        }
    }

    if(shaderDir.empty())
    {
        cout << "Could not find the Saiga shaders." << endl;
        cout << "Set the 'shaderDirectory' variable of 'SaigaParameters' accordingly." << endl;
        exit(1);
    }


    shaderPathes.addSearchPath(shaderDir);

#ifdef SAIGA_USE_VULKAN
    Vulkan::GLSLANG::init();
    Vulkan::GLSLANG::shaderPathes.addSearchPath(shaderDir);
#endif

    TextureLoader::instance()->addPath(params.textureDirectory);
    TextureLoader::instance()->addPath(".");

    Image::searchPathes.addSearchPath(".");
    Image::searchPathes.addSearchPath(params.textureDirectory);


    // Disables the following notification:
    // Buffer detailed info : Buffer object 2 (bound to GL_ELEMENT_ARRAY_BUFFER_ARB, usage hint is GL_STREAM_DRAW)
    // will use VIDEO memory as the source for buffer object operations.
    std::vector<GLuint> ignoreIds = {
        131185, //nvidia
        //intel
    };
    Error::ignoreGLError(ignoreIds);


    printSaigaInfo();
    cout<<"========================== Saiga initialization done!  =========================="<<endl;
    initialized = true;

}

void cleanupSaiga()
{
#ifdef SAIGA_USE_VULKAN
    Saiga::Vulkan::GLSLANG::quit();
#endif
    ShaderLoader::instance()->clear();
    TextureLoader::instance()->clear();
    cout<<"========================== Saiga cleanup done! =========================="<<endl;
    initialized = false;
}



}
