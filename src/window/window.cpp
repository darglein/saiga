#include "saiga/window/window.h"
#include "saiga/rendering/deferred_renderer.h"
#include "saiga/util/error.h"
#include "saiga/framework.h"
#include <cstring>
#include <FreeImagePlus.h>
#include <vector>
#include <ctime>

using std::cout;
using std::endl;


Window::Window(const std::string &name, int width, int height, bool fullscreen)
    :name(name),width(width),height(height), fullscreen(fullscreen){


}

void Window::quit(){
    cout<<"Window: Quit"<<endl;running = false;
}


bool Window::init(){

    //init window and opengl context
    if(!initWindow()){
        std::cerr<<"Failed to initialize Window!"<<std::endl;
        return false;
    }


    //inits opengl (loads functions)
    initOpenGL();

    cout<<">> Basic Window and OpenGL Context initialized!"<<endl;
    cout<<"Opengl version: "<<glGetString(GL_VERSION)<<endl;
    cout<<"GLSL version: "<<glGetString(GL_SHADING_LANGUAGE_VERSION)<<endl;
    cout<<"Renderer version: "<<glGetString(GL_RENDERER)<<endl;
    cout<<"Vendor version: "<<glGetString(GL_VENDOR)<<endl;




    if(!initInput()){
        std::cerr<<"Failed to initialize Input!"<<std::endl;
        return false;
    }

//    objLoader.materialLoader = &materialLoader;
//    materialLoader.textureLoader = &textureLoader;

#ifdef WIN32
    glDebugMessageCallback(Error::DebugLogWin32,NULL);
#else

#ifdef USE_GLEW
    glDebugMessageCallback(Error::DebugLog,NULL);
#endif
#ifdef USE_GLBINDING
     glDebugMessageCallback(Error::DebugLogWin32,NULL);
#endif
#endif

    cout<<">> Window inputs initialized!"<<endl;


    initFramework(this);

    return true;
}

void Window::resize(int width, int height)
{
    this->width = width;
    this->height = height;
    renderer->resize(width,height);
}

void Window::screenshot(const std::string &file)
{
    cout<<"Window::screenshot "<<file<<endl;
    int size = width*height*4;
    std::vector<unsigned char> data(size);


    //read data from default framebuffer and restore currently bound fb.
    GLint fb;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING,&fb);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glReadPixels(0,0,width,height,GL_RGBA,GL_UNSIGNED_BYTE,data.data());
    glBindFramebuffer(GL_FRAMEBUFFER, fb);

    for(int i = 0 ;i<size;i+=4){
        unsigned char r = data[i];
        unsigned char g = data[i+1];
        unsigned char b = data[i+2];
        unsigned char a = data[i+3];

        a = 255; //remove transparency

        // Little Endian (x86 / MS Windows, Linux) : BGR(A) order
        data[i+FI_RGBA_RED] = r;
        data[i+FI_RGBA_GREEN] = g;
        data[i+FI_RGBA_BLUE] = b;
        data[i+FI_RGBA_ALPHA] = a;
    }

    fipImage fipimg;
    fipimg.setSize(	FIT_BITMAP,width,height,32);
    auto idata = fipimg.accessPixels();
    memcpy(idata,data.data(),size);
    fipimg.save(file.c_str());
}

std::string Window::getTimeString()
{
    time_t t = time(0);   // get time now
     struct tm * now = localtime( & t );

     std::string str;
     str =     std::to_string(now->tm_year + 1900) + '-'
             + std::to_string(now->tm_mon + 1) + '-'
             + std::to_string(now->tm_mday) + '_'
             + std::to_string(now->tm_hour) + '-'
             + std::to_string(now->tm_min) + '-'
             + std::to_string(now->tm_sec);

             ;

     return str;
}

void Window::setProgram(RendererInterface *program)
{
    renderer->renderer = program;
}
