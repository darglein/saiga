
#include "window/window.h"
#include "libhello/util/error.h"
Window::Window(const std::string &name, int window_width, int window_height, bool fullscreen)
    :name(name),window_width(window_width),window_height(window_height), fullscreen(fullscreen){


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
    return true;
}

void Window::screenshot(const string &file)
{
    cout<<"Window::screenshot "<<file<<endl;
    int size = window_width*window_height*4;
    std::vector<unsigned char> data(size);


    //read data from default framebuffer and restore currently bound fb.
    GLint fb;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING,&fb);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
    glReadPixels(0,0,window_width,window_height,GL_RGBA,GL_UNSIGNED_BYTE,data.data());
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
    fipimg.setSize(	FIT_BITMAP,window_width,window_height,32);
    auto idata = fipimg.accessPixels();
    memcpy(idata,data.data(),size);
    fipimg.save(file.c_str());
}

string Window::getTimeString()
{
    time_t t = time(0);   // get time now
     struct tm * now = localtime( & t );

     string str;
     str =     std::to_string(now->tm_year + 1900) + '-'
             + std::to_string(now->tm_mon + 1) + '-'
             + std::to_string(now->tm_mday) + '_'
             + std::to_string(now->tm_hour) + '-'
             + std::to_string(now->tm_min) + '-'
             + std::to_string(now->tm_sec);

             ;

     return str;
}
