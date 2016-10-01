#include "saiga/window/window.h"
#include "saiga/rendering/deferred_renderer.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/texture/textureLoader.h"

#include "saiga/rendering/deferred_renderer.h"
#include "saiga/rendering/renderer.h"

#include "saiga/util/error.h"
#include "saiga/framework.h"
#include <cstring>
#include <vector>
#include <ctime>

using std::cout;
using std::endl;


Window::Window(const std::string &name, int width, int height)
    :name(name),width(width),height(height),
updateTimer(0.97f),interpolationTimer(0.97f),renderCPUTimer(0.97f),fpsTimer(50),upsTimer(50){

}

Window::~Window(){

    if (ssRunning){
        ssRunning = false;

        for (int i = 0; i < WRITER_COUNT; ++i){
            sswriterthreads[i]->join();
            delete sswriterthreads[i];
        }
    }


    delete renderer;
}

void Window::quit(){
    cout<<"Window: Quit"<<endl;
    running = false;
}



bool Window::init(const RenderingParameters& params){
     initSaiga();

    //init window and opengl context
    if(!initWindow()){
        std::cerr<<"Failed to initialize Window!"<<std::endl;
        return false;
    }


    //inits opengl (loads functions)
    initOpenGL();
    assert_no_glerror();

    cout<<">> Basic Window and OpenGL Context initialized!"<<endl;
    cout<<"Opengl version: "<<glGetString(GL_VERSION)<<endl;
    cout<<"GLSL version: "<<glGetString(GL_SHADING_LANGUAGE_VERSION)<<endl;
    cout<<"Renderer version: "<<glGetString(GL_RENDERER)<<endl;
    cout<<"Vendor version: "<<glGetString(GL_VENDOR)<<endl;




    if(!initInput()){
        std::cerr<<"Failed to initialize Input!"<<std::endl;
        return false;
    }


	//this somehow doesn't work in 32 bit
    glDebugMessageCallback(Error::DebugLogWin32,NULL);

    cout<<">> Window inputs initialized!"<<endl;
    assert_no_glerror();



    initDeferredRendering(params);
    assert_no_glerror();
    return true;
}

void Window::initDeferredRendering(const RenderingParameters &params)
{

    renderer = new Deferred_Renderer(getWidth(),getHeight(),params);

    renderer->lighting.loadShaders();

    PostProcessingShader* pps = ShaderLoader::instance()->load<PostProcessingShader>("post_processing.glsl"); //this shader does nothing
    std::vector<PostProcessingShader*> defaultEffects;
    defaultEffects.push_back(pps);

    renderer->postProcessor.setPostProcessingEffects(defaultEffects);

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

    int w = renderer->windowWidth;
     int h = renderer->windowHeight;

     Image img;
     img.width = w;
     img.height = h;
     img.Format() = ImageFormat(3,8,ImageElementFormat::UnsignedNormalized);
     img.create();

     //read data from default framebuffer and restore currently bound fb.
     GLint fb;
     glGetIntegerv(GL_FRAMEBUFFER_BINDING,&fb);
     glBindFramebuffer(GL_FRAMEBUFFER, 0);
     glReadPixels(0,0,w,h,GL_RGB,GL_UNSIGNED_BYTE,img.getRawData());
     glBindFramebuffer(GL_FRAMEBUFFER, fb);

     TextureLoader::instance()->saveImage(file,img);
}

void Window::screenshotRender(const std::string &file)
{
    cout<<"Window::screenshotRender "<<file<<endl;
    int w = renderer->width;
    int h = renderer->height;

    Image img;
    img.width = w;
    img.height = h;
    img.Format() = ImageFormat(3,8,ImageElementFormat::UnsignedNormalized);
    img.create();

    auto tex = getRenderer()->postProcessor.getCurrentTexture();
    tex->bind();
    glGetTexImage(tex->getTarget(),0,GL_RGB,GL_UNSIGNED_BYTE,img.getRawData());
    tex->unbind();

    TextureLoader::instance()->saveImage(file,img);
}

void Window::screenshotParallelWrite(const std::string &file){

    if (currentScreenshot == 0){
        cout<<"Starting " << WRITER_COUNT << " screenshot writers" <<file<<endl;
        for (int i = 0; i < WRITER_COUNT; ++i){
            sswriterthreads[i] = new std::thread(&Window::processScreenshots, this);
        }
        ssRunning = true;
    }


    int w = renderer->width;
    int h = renderer->height;

    std::shared_ptr<Image> img = std::make_shared<Image>();
    img->width = w;
    img->height = h;
    img->Format() = ImageFormat(3,8,ImageElementFormat::UnsignedNormalized);
    img->create();

    auto tex = getRenderer()->postProcessor.getCurrentTexture();
    tex->bind();
    glGetTexImage(tex->getTarget(),0,GL_RGB,GL_UNSIGNED_BYTE,img->getRawData());
    tex->unbind();



    if (waitForWriters){

        while(true){
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            lock.lock();
            if (queue.size() < 5){
                lock.unlock();
                break;
            }
            lock.unlock();
        }

        waitForWriters = false;
    }
    lock.lock();
    parallelScreenshotPath = file;
    queue.push_back(img);


    if ((int)queue.size() > queueLimit){ //one frame full HD ~ 4.5Mb
        waitForWriters = true;
    }
//        cout << "queue size: " << queue.size() << endl;

    lock.unlock();
}


void Window::processScreenshots()
{

    while(ssRunning){
        int cur = 0;
        bool took = false;
        int queueSize = 0;
        lock.lock();
        std::shared_ptr<Image> f;
        if (!queue.empty()){
            f = queue.front();
            queueSize = queue.size();
            if (f){
                took = true;
                queue.pop_front();
                cur = currentScreenshot++;
            }
        }

        lock.unlock();

        if (took){
            long long start = std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
            TextureLoader::instance()->saveImage(parallelScreenshotPath+ std::to_string(cur) + ".bmp",*f);
//            f->save(().c_str());
            cout << "write " << cur  << " (" <<queueSize << ") " << (std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now().time_since_epoch()).count() - start)/1000 << "ms"<< endl;


        } else {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }
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

Ray Window::createPixelRay(const glm::vec2 &pixel) const
{
    vec4 p = vec4(2*pixel.x/Window::width-1.f,1.f-(2*pixel.y/Window::height),0,1.f);
    p = glm::inverse(Window::currentCamera->proj)*p;
    p /= p.w;

    mat4 inverseView = glm::inverse(Window::currentCamera->view);
    vec3 ray_world =vec3(inverseView*p);
    vec3 origin = vec3(inverseView[3]);
    return Ray(glm::normalize(ray_world-origin),origin);
}

Ray Window::createPixelRay(const glm::vec2 &pixel, const vec2& resolution, const mat4& inverseProj) const
{
    vec4 p = vec4(2*pixel.x/resolution.x-1.f,1.f-(2*pixel.y/resolution.y),0,1.f);
    p = inverseProj*p;
    p /= p.w;

    mat4 inverseView = glm::inverse(Window::currentCamera->view);
    vec3 ray_world =vec3(inverseView*p);
    vec3 origin = vec3(inverseView[3]);
    return Ray(glm::normalize(ray_world-origin),origin);
}

vec3 Window::screenToWorld(const glm::vec2 &pixel) const
{
    vec4 p = vec4(2*pixel.x/Window::width-1.f,1.f-(2*pixel.y/Window::height),0,1.f);
    p = glm::inverse(Window::currentCamera->proj)*p;
    p /= p.w;

    mat4 inverseView = glm::inverse(Window::currentCamera->view);
    vec3 ray_world =vec3(inverseView*p);
    return ray_world;
}


vec3 Window::screenToWorld(const glm::vec2 &pixel, const vec2& resolution, const mat4& inverseProj) const
{
    vec4 p = vec4(2*pixel.x/resolution.x-1.f,1.f-(2*pixel.y/resolution.y),0,1.f);
    p = inverseProj*p;
    p /= p.w;

    mat4 inverseView = glm::inverse(Window::currentCamera->view);
    vec3 ray_world =vec3(inverseView*p);
    return ray_world;
}



vec2 Window::projectToScreen(const glm::vec3 &pos) const
{
    vec4 r = Window::currentCamera->proj * Window::currentCamera->view * vec4(pos,1);
    r /= r.w;

    vec2 pixel;
    pixel.x = (r.x +1.f)*Window::width *0.5f;
    pixel.y = -(r.y - 1.f) * Window::height * 0.5f;

    return pixel;
}

void Window::update(float dt)
{
    updateTimer.start();
    renderer->renderer->update(dt);
    updateTimer.stop();

    upsTimer.stop();
    upsTimer.start();
}

void Window::render(float dt, float interpolation)
{
    interpolationTimer.start();
    renderer->renderer->interpolate(dt,interpolation);
    interpolationTimer.stop();

    renderCPUTimer.start();
    renderer->render_intern();
    renderCPUTimer.stop();

    fpsTimer.stop();
    fpsTimer.start();
}

tick_t Window::getGameTicks(){
    gameTimer.stop();
    return gameTimer.getTimeMicrS();
}

void Window::sleep(tick_t ticks)
{
    if(ticks > 0){
        std::this_thread::sleep_for(std::chrono::microseconds(ticks));
//        std::cout << "Sleeping " << ticks << " ticks (" << ((float)ticks/getGameTicksPerSecond()) << " seconds)" << std::endl;
    }
}

void Window::setTimeScale(double timeScale)
{
    this->timeScale = timeScale;
}



void Window::startMainLoop(int updatesPerSecond, int framesPerSecond, int maxFrameSkip)
{
    gameTimer.start();

    if(updatesPerSecond <= 0)
        updatesPerSecond = getGameTicksPerSecond();
    if(framesPerSecond <= 0)
        framesPerSecond = getGameTicksPerSecond();


    float updateDT = 1.0f / updatesPerSecond;
    float framesDT = 1.0f / framesPerSecond;

    tick_t ticksPerUpdate = getGameTicksPerSecond() / updatesPerSecond;
    tick_t ticksPerFrame = getGameTicksPerSecond() / framesPerSecond;


    tick_t nextUpdateTick = getGameTicks();
    tick_t nextFrameTick = nextUpdateTick;

    while(!shouldClose()){

        checkEvents();

        //With this loop we are able to skip frames if the system can't keep up.
        for(int i = 0; i <= maxFrameSkip && getGameTicks() > nextUpdateTick; ++i){
            update(updateDT);
            nextUpdateTick += ticksPerUpdate / timeScale;
        }

        if(getGameTicks() > nextFrameTick){
            //calculate the interpolation value. Usefull when the framerate is higher than the update rate
            tick_t lastUpdate = nextUpdateTick - ticksPerUpdate;
            tick_t ticksSinceLastUpdate = getGameTicks() - lastUpdate;
            float interpolation = ticksSinceLastUpdate / (float)ticksPerUpdate;
            interpolation = glm::clamp(interpolation,0.0f,1.0f);

            render(updateDT,interpolation);
            swapBuffers();
            nextFrameTick += ticksPerFrame;
        }

        //sleep until the next interesting event
        tick_t nextEvent = glm::min(nextFrameTick,nextUpdateTick);
        sleep(nextEvent - getGameTicks());
        assert_no_glerror_end_frame();
    }
}
