#include "saiga/window/window.h"
#include "saiga/rendering/deferred_renderer.h"
#include "saiga/opengl/shader/shaderLoader.h"

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

    initDeferredRendering();

    return true;
}

void Window::initDeferredRendering()
{
    //     exit(1);

    renderer = new Deferred_Renderer();
    renderer->init(getWidth(),getHeight());
    //    renderer->lighting.setShader(shaderLoader.load<SpotLightShader>("deferred_lighting_spotlight.glsl"));


    renderer->lighting.loadShaders();


    //    renderer->postProcessingShader  = shaderLoader.load<PostProcessingShader>("fxaa.glsl");
    //    renderer->postProcessingShader  = shaderLoader.load<PostProcessingShader>("SMAA.glsl");
    //    renderer->postProcessingShader  = shaderLoader.load<PostProcessingShader>("gaussian_blur.glsl");


    renderer->ssaoShader  =  ShaderLoader::instance()->load<SSAOShader>("ssao.glsl");
    renderer->ssao = true;
    //    renderer->otherShader  =  ShaderLoader::instance()->load<PostProcessingShader>("post_processing.glsl");

    PostProcessingShader* pps = ShaderLoader::instance()->load<PostProcessingShader>("post_processing.glsl");
    std::vector<PostProcessingShader*> defaultEffects;
    defaultEffects.push_back(pps);

    renderer->postProcessor.setPostProcessingEffects(defaultEffects);
    //    renderer->postProcessor.postProcessingEffects.push_back(renderer->postProcessingShader);

    //    PostProcessingShader* bla = ShaderLoader::instance()->load<PostProcessingShader>("gaussian_blur.glsl");
    //    renderer->postProcessor.postProcessingEffects.push_back(bla);
    //    renderer->postProcessor.postProcessingEffects.push_back(bla);
    //    renderer->postProcessor.postProcessingEffects.push_back(bla);
    //    renderer->postProcessor.postProcessingEffects.push_back(bla);

    renderer->lighting.setRenderDebug( false);

    renderer->currentCamera = &this->currentCamera;

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

void Window::setProgram(Program *program)
{
    renderer->renderer = program;
}

Ray Window::createPixelRay(const glm::vec2 &pixel)
{
    vec4 p = vec4(2*pixel.x/Window::width-1.f,1.f-(2*pixel.y/Window::height),0,1.f);
    p = glm::inverse(Window::currentCamera->proj)*p;
    p /= p.w;

    mat4 inverseView = glm::inverse(Window::currentCamera->view);
    vec3 ray_world =vec3(inverseView*p);
    vec3 origin = vec3(inverseView[3]);
    return Ray(glm::normalize(ray_world-origin),origin);
}
