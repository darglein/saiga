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
#include <thread>

using std::cout;
using std::endl;


void WindowParameters::setMode(bool fullscreen, bool borderLess)
{
    if(fullscreen){
        mode = (borderLess) ? Mode::borderLessFullscreen : Mode::fullscreen;
    }else{
        mode = (borderLess) ? Mode::borderLessWindowed : Mode::windowed;
    }
}

Window::Window(WindowParameters _windowParameters)
    :windowParameters(_windowParameters),
      updateTimer(0.97f),interpolationTimer(0.97f),renderCPUTimer(0.97f),swapBuffersTimer(0.97f),fpsTimer(50),upsTimer(50){

}

Window::~Window(){
    delete renderer;
}

void Window::close(){
    cout<<"Window: close"<<endl;
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


    if(!initInput()){
        std::cerr<<"Failed to initialize Input!"<<std::endl;
        return false;
    }


    //this somehow doesn't work in 32 bit windows
    //on older linux system the last parameter of the function is void* instead of const void*
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

    PostProcessingShader* pps = ShaderLoader::instance()->load<PostProcessingShader>("post_processing/post_processing.glsl"); //this shader does nothing
    std::vector<PostProcessingShader*> defaultEffects;
    defaultEffects.push_back(pps);

    renderer->postProcessor.setPostProcessingEffects(defaultEffects);

    renderer->lighting.setRenderDebug( false);

    renderer->currentCamera = &this->currentCamera;

}

void Window::resize(int width, int height)
{
    this->windowParameters.width = width;
    this->windowParameters.height = height;
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

Ray Window::createPixelRay(const vec2 &pixel) const
{
    vec4 p = vec4(2*pixel.x/getWidth()-1.f,1.f-(2*pixel.y/getHeight()),0,1.f);
    p = glm::inverse(Window::currentCamera->proj)*p;
    p /= p.w;

    mat4 inverseView = glm::inverse(Window::currentCamera->view);
    vec3 ray_world =vec3(inverseView*p);
    vec3 origin = vec3(inverseView[3]);
    return Ray(glm::normalize(ray_world-origin),origin);
}

Ray Window::createPixelRay(const vec2 &pixel, const vec2& resolution, const mat4& inverseProj) const
{
    vec4 p = vec4(2*pixel.x/resolution.x-1.f,1.f-(2*pixel.y/resolution.y),0,1.f);
    p = inverseProj*p;
    p /= p.w;

    mat4 inverseView = glm::inverse(Window::currentCamera->view);
    vec3 ray_world =vec3(inverseView*p);
    vec3 origin = vec3(inverseView[3]);
    return Ray(glm::normalize(ray_world-origin),origin);
}

vec3 Window::screenToWorld(const vec2 &pixel) const
{
    vec4 p = vec4(2*pixel.x/getWidth()-1.f,1.f-(2*pixel.y/getHeight()),0,1.f);
    p = glm::inverse(Window::currentCamera->proj)*p;
    p /= p.w;

    mat4 inverseView = glm::inverse(Window::currentCamera->view);
    vec3 ray_world =vec3(inverseView*p);
    return ray_world;
}


vec3 Window::screenToWorld(const vec2 &pixel, const vec2& resolution, const mat4& inverseProj) const
{
    vec4 p = vec4(2*pixel.x/resolution.x-1.f,1.f-(2*pixel.y/resolution.y),0,1.f);
    p = inverseProj*p;
    p /= p.w;

    mat4 inverseView = glm::inverse(Window::currentCamera->view);
    vec3 ray_world =vec3(inverseView*p);
    return ray_world;
}



vec2 Window::projectToScreen(const vec3 &pos) const
{
    vec4 r = Window::currentCamera->proj * Window::currentCamera->view * vec4(pos,1);
    r /= r.w;

    vec2 pixel;
    pixel.x = (r.x +1.f)*getWidth() *0.5f;
    pixel.y = -(r.y - 1.f) * getHeight() * 0.5f;

    return pixel;
}

void Window::update(float dt)
{
    updateTimer.start();
    endParallelUpdate();
    renderer->renderer->update(dt);
    startParallelUpdate(dt);
    updateTimer.stop();

    numUpdates++;

    upsTimer.stop();
    upsTimer.start();
}




void Window::startParallelUpdate(float dt)
{

    if(parallelUpdate){
        semStartUpdate.notify();
    }else{
        parallelUpdateCaller(dt);
    }
}

void Window::endParallelUpdate()
{
    if(parallelUpdate)
        semFinishUpdate.wait();
}

void Window::parallelUpdateThread(float dt)
{
    semFinishUpdate.notify();
    semStartUpdate.wait();
    while(running){
            parallelUpdateCaller(dt);
            semFinishUpdate.notify();
            semStartUpdate.wait();
    }
}

void Window::parallelUpdateCaller(float dt)
{
    renderer->renderer->parallelUpdate(dt);
}

void Window::render(float dt, float interpolation)
{
    interpolationTimer.start();
    renderer->renderer->interpolate(dt,interpolation);
    interpolationTimer.stop();

    renderCPUTimer.start();
    renderer->render_intern();
    renderCPUTimer.stop();

    numFrames++;

    swapBuffersTimer.start();
    swapBuffers();
    swapBuffersTimer.stop();

    fpsTimer.stop();
    fpsTimer.start();
}



tick_t Window::getGameTicks(){
    gameTimer.stop();
    return gameTimer.getTime() - gameLoopDelay;
}

void Window::sleep(tick_t ticks)
{
    if(ticks > tick_t(0)){
        std::this_thread::sleep_for(ticks);
    }
}

void Window::setTimeScale(double timeScale)
{
    this->timeScale = timeScale;
}



void Window::startMainLoop(int updatesPerSecond, int framesPerSecond, float mainLoopInfoTime, int maxFrameSkip, bool _parallelUpdate, bool catchUp)
{
    parallelUpdate = _parallelUpdate;
    gameTimer.start();

    cout << "> Starting the main loop..." << endl;
    cout << "> updatesPerSecond=" << updatesPerSecond << " framesPerSecond=" << framesPerSecond << " maxFrameSkip=" << maxFrameSkip << endl;



    if(updatesPerSecond <= 0)
        updatesPerSecond = gameTime.base.count();
    if(framesPerSecond <= 0)
        framesPerSecond = gameTime.base.count();


    float updateDT = 1.0f / updatesPerSecond;
    //    float framesDT = 1.0f / framesPerSecond;

    tick_t ticksPerUpdate = gameTime.base / updatesPerSecond;
    tick_t ticksPerFrame = gameTime.base / framesPerSecond;
    tick_t ticksPerInfo = std::chrono::duration_cast<tick_t>(gameTime.base * mainLoopInfoTime);

    gameTime.dt = ticksPerUpdate;
    gameTime.dtr = ticksPerFrame;


    tick_t nextUpdateTick = getGameTicks();
    tick_t lastUpdateTick = nextUpdateTick;
    tick_t nextFrameTick = nextUpdateTick;
    tick_t nextInfoTick = nextUpdateTick;

    tick_t actualUpdateTick = tick_t(0);
    tick_t actualFrameTick = tick_t(0);

    tick_t maxGameLoopDelay = std::chrono::duration_cast<tick_t>(std::chrono::milliseconds(1000));

    if(parallelUpdate){
        updateThread = std::thread(&Window::parallelUpdateThread,this,updateDT);
    }


    while(true){

        tick_t currentTicksPerUpdate = std::chrono::duration_cast<tick_t>(ticksPerUpdate / timeScale);
        checkEvents();

        if(shouldClose()){
            break;
        }

        //With this loop we are able to skip frames if the system can't keep up.
        for(int i = 0; i <= maxFrameSkip && getGameTicks() > nextUpdateTick; ++i){
            actualUpdateTick = getGameTicks();

            gameTime.time = nextUpdateTick;
            gameTime.updatetime = gameTime.time;

            if(actualUpdateTick - gameTime.updatetime > maxGameLoopDelay){
                if(catchUp == false){
                    tick_t newDelay = actualUpdateTick - gameTime.updatetime;
                    gameLoopDelay += newDelay;
//                     cout << "> Warning: Cannot keep up with the set update rate. Adding delay of "
//                          << std::chrono::duration_cast<std::chrono::duration<float,std::milli>>(newDelay).count() << "ms" << endl;
                     gameloopDropAccumulatedUpdates = true;
                }
            }


            update(updateDT);

            if (gameloopDropAccumulatedUpdates){
//                cout << "> Advancing game loop to live." << endl;
                gameTime.updatetime = getGameTicks();
                nextFrameTick = nextUpdateTick;
                gameloopDropAccumulatedUpdates = false;
            }


            lastUpdateTick = gameTime.updatetime;
            nextUpdateTick = gameTime.updatetime + currentTicksPerUpdate;
        }

        if(getGameTicks() > nextFrameTick){
            actualFrameTick = getGameTicks();

            tick_t ticksSinceLastUpdate = actualFrameTick - actualUpdateTick;

           gameTime.time = gameTime.updatetime + ticksSinceLastUpdate;


            //calculate the interpolation value. Useful when the framerate is higher than the update rate
            float interpolation = (float)ticksSinceLastUpdate.count() / (nextUpdateTick - lastUpdateTick).count();
            interpolation = glm::clamp(interpolation,0.0f,1.0f);

            render(updateDT,interpolation);
            nextFrameTick = nextFrameTick + ticksPerFrame;
        }

        if(getGameTicks() > nextInfoTick){
            auto gt = std::chrono::duration_cast<std::chrono::seconds>(getGameTicks());
            cout << "> Time: " << gt.count() << "s  Total number of updates/frames: " << numUpdates << "/" << numFrames << "  UPS/FPS: " << (1000.0f/upsTimer.getTimeMS()) << "/" << (1000.0f/fpsTimer.getTimeMS()) << endl;
            nextInfoTick += ticksPerInfo;
        }

        //sleep until the next interesting event
        tick_t nextEvent = nextFrameTick < nextUpdateTick ? nextFrameTick : nextUpdateTick;
        sleep(nextEvent - getGameTicks());
        assert_no_glerror_end_frame();
    }
    running = false;

    if(parallelUpdate){
        //cleanup the update thread
        cout << "Finished main loop. Exiting update thread." << endl;
        endParallelUpdate();
        semStartUpdate.notify();
        updateThread.join();
    }

    auto gt = std::chrono::duration_cast<std::chrono::seconds>(getGameTicks());
    cout << "> Main loop finished in " << gt.count() << "s  Total number of updates/frames: " << numUpdates << "/" << numFrames  << endl;
}
