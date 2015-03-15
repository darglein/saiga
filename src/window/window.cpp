
#include "window/window.h"
#include "libhello/util/error.h"
Window::Window(const std::string &name, int window_width, int window_height)
    :name(name),window_width(window_width),window_height(window_height){

    objLoader.addPath(".");
    materialLoader.addPath(".");
}


bool Window::init(){
    if(!initWindow()){
        std::cerr<<"Failed to initialize Window!"<<std::endl;
        return false;
    }

    cout<<">> Basic Window and OpenGL Context initialized!"<<endl;
    cout<<"Opengl version: "<<glGetString(GL_VERSION)<<endl;
    cout<<"GLSL version: "<<glGetString(GL_SHADING_LANGUAGE_VERSION)<<endl;
    cout<<"Renderer version: "<<glGetString(GL_RENDERER)<<endl;
    cout<<"Vendor version: "<<glGetString(GL_VENDOR)<<endl;

    if(!initInput()){
        std::cerr<<"Failed to initialize Input!"<<std::endl;
        return false;
    }

    objLoader.materialLoader = &materialLoader;
    materialLoader.textureLoader = &textureLoader;

#ifdef WIN32
    glDebugMessageCallback(Error::DebugLogWin32,NULL);
#else
    glDebugMessageCallback(Error::DebugLog,NULL);
#endif

    cout<<">> Window inputs initialized!"<<endl;
    return true;
}
