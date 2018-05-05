/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/framework.h"

#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/texture/textureLoader.h"
#include "saiga/opengl/error.h"
#include "saiga/util/configloader.h"
#include "saiga/util/assert.h"

#include "saiga/cuda/tests/test.h"
#include "saiga/util/floatingPoint.h"

namespace Saiga {

bool initialized = false;

std::string SHADER_PATH;
std::string TEXTURE_PATH;
std::string MATERIAL_PATH;
std::string OBJ_PATH;

void readConfigFile(){
    ConfigLoader cl;
    cl.loadFile2("saiga-config.txt");

    SHADER_PATH = cl.getString("SHADER_PATH","/usr/local/share/saiga/shader");
    TEXTURE_PATH = cl.getString("TEXTURE_PATH","textures");

    cl.writeFile();

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
#ifdef SAIGA_USE_GLBINDING
    libs += "GLBINDING,";
#endif
#ifdef SAIGA_USE_GLEW
    libs += "GLEW,";
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

void writeExtensions(){


    std::ofstream myfile;
    myfile.open ("opengl-extensions.txt");



    int n = getExtensionCount();
    for (GLint i=0; i<n; i++)
    {
        const char* extension = (const char*) glGetStringi(GL_EXTENSIONS, i);

        myfile << extension<<endl;
    }

    myfile.close();
}

void initSaiga()
{
    SAIGA_ASSERT(!initialized);

    FP::resetSSECSR();

    readConfigFile();


    shaderPathes.addSearchPath(SHADER_PATH);

    TextureLoader::instance()->addPath(TEXTURE_PATH);
    TextureLoader::instance()->addPath(OBJ_PATH);
    TextureLoader::instance()->addPath(".");
	 
	// Disables the following notification:
	// Buffer detailed info : Buffer object 2 (bound to GL_ELEMENT_ARRAY_BUFFER_ARB, usage hint is GL_STREAM_DRAW) 
	// will use VIDEO memory as the source for buffer object operations.
	std::vector<GLuint> ignoreIds = { 131185 };
	Error::ignoreGLError(ignoreIds);


    printSaigaInfo();
    cout<<"========================== Saiga initialization done!  =========================="<<endl;
    initialized = true;

}

void cleanupSaiga()
{
    SAIGA_ASSERT(initialized);

    ShaderLoader::instance()->clear();
    TextureLoader::instance()->clear();
    cout<<"========================== Saiga cleanup done! =========================="<<endl;
    initialized = false;
}


}
