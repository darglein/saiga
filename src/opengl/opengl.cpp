#include "saiga/opengl/opengl.h"

#include <algorithm>
#include "saiga/util/assert.h"

namespace Saiga {

bool openglinitialized = false;

void initOpenGL()
{
    SAIGA_ASSERT(!openglinitialized);
#ifdef SAIGA_USE_GLEW
    //Initialize GLEW
    glewExperimental = GL_TRUE;
    GLenum glewError = glewInit();
    if( glewError != GLEW_OK ){
        std::cerr<<"Error initializing GLEW! "<< glewGetErrorString( glewError ) <<std::endl;
        SAIGA_ASSERT(0);
    }
    glGetError(); //ignore first gl error after glew init
#endif

#ifdef SAIGA_USE_GLBINDING
    glbinding::Binding::initialize();
#endif
    openglinitialized = true;
    std::cout << "> OpenGL initialized" << std::endl;
    printOpenGLVersion();
}

void printOpenGLVersion() {
    std::cout << "Opengl version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "GLSL version: " << glGetString(GL_SHADING_LANGUAGE_VERSION) << std::endl;
    std::cout << "Renderer: " << glGetString(GL_RENDERER) << std::endl;
    std::cout << "Vendor: " << glGetString(GL_VENDOR) << " - ";

    switch(getOpenGLVendor()){
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

bool OpenGLisInitialized(){
    return openglinitialized;
}

int getVersionMajor(){
    int v;
    glGetIntegerv(GL_MAJOR_VERSION,&v);
    return v;
}

int getVersionMinor(){
    int v;
    glGetIntegerv(GL_MINOR_VERSION,&v);
    return v;
}


int getExtensionCount(){
    GLint n=0;
    glGetIntegerv(GL_NUM_EXTENSIONS, &n);
    return n;
}

bool hasExtension(const std::string &ext){
    int n = getExtensionCount();
    for (GLint i=0; i<n; i++)
    {
        const char* extension = (const char*) glGetStringi(GL_EXTENSIONS, i);
        if(ext == std::string(extension)){
            return true;
        }
    }
    return false;
}

OpenGLVendor getOpenGLVendor()
{
    std::string ven = reinterpret_cast<const char*>(glGetString(GL_VENDOR));
    std::transform(ven.begin(), ven.end(), ven.begin(), ::tolower);


    if (ven.find("nvidia") != std::string::npos) {
        return OpenGLVendor::Nvidia;
    }

    if (ven.find("ati") != std::string::npos) {
        return OpenGLVendor::Ati;
    }


    if (ven.find("intel") != std::string::npos) {
        return OpenGLVendor::Intel;
    }


    if (ven.find("mesa") != std::string::npos) {
        return OpenGLVendor::Mesa;
    }


    return OpenGLVendor::Unknown;
}

}
