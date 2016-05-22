#include "saiga/opengl/opengl.h"
#include <iostream>
#include "saiga/util/assert.h"

bool openglinitialized = false;

#ifdef USE_GLEW
void initOpenGL()
{
	assert(!openglinitialized);

    //Initialize GLEW
    glewExperimental = GL_TRUE;
    GLenum glewError = glewInit();
    if( glewError != GLEW_OK ){
        std::cerr<<"Error initializing GLEW! "<< glewGetErrorString( glewError ) <<std::endl;
        assert(0);
    }

    glGetError(); //ignore first gl error after glew init

    openglinitialized = true;
}
#endif

#ifdef USE_GLBINDING
void initOpenGL()
{
	assert(!openglinitialized);
    glbinding::Binding::initialize();
    openglinitialized = true;
}
#endif


void terminateOpenGL()
{
	assert(openglinitialized);
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
