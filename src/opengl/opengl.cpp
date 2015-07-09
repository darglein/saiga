#include "saiga/opengl/opengl.h"
#include <iostream>

#ifdef USE_GLEW
void initOpenGL()
{
    //Initialize GLEW
    glewExperimental = GL_TRUE;
    GLenum glewError = glewInit();
    if( glewError != GLEW_OK ){
        std::cerr<<"Error initializing GLEW! "<< glewGetErrorString( glewError ) <<std::endl;
        std::exit(1);
    }

    glGetError(); //ignore first gl error after glew init
}
#endif

#ifdef USE_GLBINDING
void initOpenGL()
{
    glbinding::Binding::initialize();
}
#endif
