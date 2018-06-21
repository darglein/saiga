/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/window/window.h"
#include "saiga/rendering/deferred_renderer.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/texture/textureLoader.h"
#include "saiga/image/glImageFormat.h"
#include "saiga/rendering/deferred_renderer.h"
#include "saiga/rendering/renderer.h"
#include "saiga/rendering/program.h"

#include "saiga/util/tostring.h"
#include "saiga/opengl/error.h"
#include "saiga/framework.h"
#include "saiga/imgui/imgui.h"
#include "saiga/util/math.h"

#include <cstring>
#include <vector>
#include <ctime>
#include <thread>

namespace Saiga {



OpenGLWindow::OpenGLWindow(WindowParameters _windowParameters)
    :windowParameters(_windowParameters),
      updateTimer(0.97f),interpolationTimer(0.97f),renderCPUTimer(0.97f),swapBuffersTimer(0.97f),fpsTimer(50),upsTimer(50){
    memset(imUpdateTimes, 0, numGraphValues * sizeof(float));
    memset(imRenderTimes, 0, numGraphValues * sizeof(float));
}

OpenGLWindow::~OpenGLWindow(){
//    delete renderer;
}

void OpenGLWindow::close(){
    cout<<"Window: close"<<endl;
    running = false;
}

void OpenGLWindow::updateUpdateGraph()
{
    ut = std::chrono::duration<double, std::milli>(updateTimer.getTime()).count();

    maxUpdateTime = std::max(ut,maxUpdateTime);

    avUt = 0;
    for(int i = 0 ;i < numGraphValues; ++i){
        avUt +=   imUpdateTimes[i];
    }
    avUt /= numGraphValues;

    imUpdateTimes[imCurrentIndexUpdate] = ut;
    imCurrentIndexUpdate = (imCurrentIndexUpdate+1) % numGraphValues;
}

void OpenGLWindow::updateRenderGraph()
{
    //    ft = renderer->getUnsmoothedTimeMS(Deferred_Renderer::DeferredTimings::TOTAL);
    if(renderer)
        ft = renderer->getTotalRenderTime();
    maxRenderTime = std::max(ft,maxRenderTime);

    avFt = 0;
    for(int i = 0 ;i < numGraphValues; ++i){
        avFt +=   imRenderTimes[i];
    }
    avFt /= numGraphValues;
    imRenderTimes[imCurrentIndexRender] = ft;
    imCurrentIndexRender = (imCurrentIndexRender+1) % numGraphValues;
}


void OpenGLWindow::renderImGui(bool *p_open)
{
    if(!showImgui)
        return;

    p_open = &showImgui;

    int w = 340;
    int h = 240;
    ImGui::SetNextWindowPos(ImVec2(0, getHeight() - h), ImGuiSetCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(w,h), ImGuiSetCond_FirstUseEver);
    ImGui::Begin("OpenGLWindow",&showImgui);





    ImGui::Text("Update Time: %fms Ups: %f",ut, 1000.0f / upsTimer.getTimeMS());
    ImGui::PlotLines("Update Time", imUpdateTimes, numGraphValues, imCurrentIndexUpdate, ("avg "+Saiga::to_string(avUt)).c_str(), 0,maxUpdateTime, ImVec2(0,80));
    ImGui::Text("Render Time: %fms Fps: %f",ft, 1000.0f / fpsTimer.getTimeMS());
    ImGui::PlotLines("Render Time", imRenderTimes, numGraphValues, imCurrentIndexRender, ("avg "+Saiga::to_string(avFt)).c_str(), 0,maxRenderTime, ImVec2(0,80));
    if(ImGui::Button("Reset Max Value"))
    {
        maxUpdateTime = 1;
        maxRenderTime = 1;
    }


    ImGui::Text("Swap Time: %fms", swapBuffersTimer.getTimeMS());
    ImGui::Text("Interpolate Time: %fms", interpolationTimer.getTimeMS());
    ImGui::Text("Render CPU Time: %fms", renderCPUTimer.getTimeMS());

    ImGui::Text("Running: %d",running);
    ImGui::Text("numUpdates: %d",numUpdates);
    ImGui::Text("numFrames: %d",numFrames);

    std::chrono::duration<double, std::milli> dt = gameTime.dt;
    ImGui::Text("Timestep: %fms",dt.count());

    std::chrono::duration<double, std::milli> delay = gameLoopDelay;
    ImGui::Text("Delay: %fms",delay.count());

    float scale = gameTime.getTimeScale();
    ImGui::SliderFloat("Time Scale",&scale,0,5);
    gameTime.setTimeScale(scale);

    ImGui::Text("Camera Position: %s" , to_string(currentCamera->getPosition()).c_str());
    ImGui::Text("Camera Direction: %s" , to_string(-vec3(currentCamera->getDirection())).c_str());
    if(ImGui::Button("Printf camera"))
    {
        cout << "camera.position = vec4" << currentCamera->position << ";" << endl;
        cout << "camera.rot = quat" << currentCamera->rot << ";" << endl;
        //        createTRSmatrix()
    }

    if(ImGui::Button("Reload Shaders")){
        ShaderLoader::instance()->reload();
    }

    ImGui::Checkbox("showRendererImgui",&showRendererImgui);
    ImGui::Checkbox("showImguiDemo",&showImguiDemo);

    ImGui::End();

    if(showRendererImgui && renderer)
    {
        renderer->renderImGui(&showRendererImgui);
    }


    if(showImguiDemo){
        ImGui::SetNextWindowPos(ImVec2(340, 0), ImGuiSetCond_FirstUseEver);
        ImGui::ShowTestWindow(&showImguiDemo);
    }
}

bool OpenGLWindow::create()
{
    initSaiga(windowParameters.saigaParameters);

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


    //in older glew versions the last parameter of the function is void* instead of const void*
#if defined(GLEW_VERSION_4_5) || defined(SAIGA_USE_GLBINDING)

    //this somehow doesn't work on windows 32 bit
#if !defined _WIN32 || defined _WIN64
    glDebugMessageCallback(Error::DebugLogConst,NULL);
#endif

#else
    glDebugMessageCallback(Error::DebugLog,NULL);
#endif

    cout<<">> Window inputs initialized!"<<endl;
    assert_no_glerror();

    return true;

}

void OpenGLWindow::destroy()
{
    terminateOpenGL();
    cleanupSaiga();
    freeContext();
}




void OpenGLWindow::resize(int width, int height)
{
    this->windowParameters.width = width;
    this->windowParameters.height = height;
    renderer->resize(width,height);
}


void OpenGLWindow::readToExistingImage(Image &out)
{
    //read data from default framebuffer and restore currently bound fb.
    GLint fb;
    glGetIntegerv(GL_FRAMEBUFFER_BINDING,&fb);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);


    //    glReadPixels(0,0,out.width,out.height,GL_RGB,GL_UNSIGNED_BYTE,out.getRawData());

    //    SAIGA_ASSERT(0);
    //    glReadPixels(0,0,out.width,out.height,out.Format().getGlFormat(),out.Format().getGlType(),out.getRawData());
    glReadPixels(0,0,out.width,out.height,getGlFormat(out.type),getGlType(out.type),out.data());


    glBindFramebuffer(GL_FRAMEBUFFER, fb);
}


void OpenGLWindow::readToImage(Image& out){
    int w = renderer->outputWidth;
    int h = renderer->outputHeight;

    //    out.width = w;
    //    out.height = h;
    out.create(h,w,UC3);
    //    out.Format() = ImageFormat(3,8,ImageElementFormat::UnsignedNormalized);
    //    SAIGA_ASSERT(0);
    //    out.create();

    readToExistingImage(out);
}


void OpenGLWindow::screenshot(const std::string &file)
{
    Image img;
    readToImage(img);
    img.save(file);
    //    TextureLoader::instance()->saveImage(file,img);
}

void OpenGLWindow::screenshotRender(const std::string &file)
{
    SAIGA_ASSERT(0);
#if 0
    //    cout<<"Window::screenshotRender "<<file<<endl;
    int w = renderer->width;
    int h = renderer->height;

    Image img;
    img.width = w;
    img.height = h;
    //    img.Format() = ImageFormat(3,8,ImageElementFormat::UnsignedNormalized);
    SAIGA_ASSERT(0);
    img.create();

    auto tex = getRenderer()->postProcessor.getCurrentTexture();
    tex->bind();
    glGetTexImage(tex->getTarget(),0,GL_RGB,GL_UNSIGNED_BYTE,img.data());
    tex->unbind();

    //    TextureLoader::instance()->saveImage(file,img);
    img.save(file);
#endif
}

void OpenGLWindow::getDepthFloat(Image& out){
    SAIGA_ASSERT(0);
#if 0
    int w = renderer->outputWidth;
    int h = renderer->outputHeight;

    out.width = w;
    out.height = h;
    //    out.Format() = ImageFormat(1,32,ImageElementFormat::FloatingPoint);
    SAIGA_ASSERT(0);
    out.create();


    Image img;
    img.width = w;
    img.height = h;
    //    img.Format() = ImageFormat(4,8,ImageElementFormat::UnsignedNormalized);
    SAIGA_ASSERT(0);
    img.create();


    auto tex = getRenderer()->gbuffer.getTextureDepth();
    tex->bind();
    glGetTexImage(tex->getTarget(),0,GL_DEPTH_STENCIL,GL_UNSIGNED_INT_24_8,img.data());
    tex->unbind();

    for(int i = 0; i < h; ++i)
    {
        for(int j = 0; j < w; ++j)
        {
#if 0
            unsigned int v = img.getPixel<unsigned int>(j,i);
            //override stencil bits with 0
            v = v & 0xFFFFFF00;
            float d = uintToNFloat(v);
            out.getPixel<float>(j,i) = d;
#endif
        }
    }
#endif

}

void OpenGLWindow::screenshotRenderDepth(const std::string &file)
{
    SAIGA_ASSERT(0);
#if 0
    //    cout<<"Window::screenshotRender "<<file<<endl;
    int w = renderer->width;
    int h = renderer->height;

    Image img;
    getDepthFloat(img);

    Image img2;
    img2.width = w;
    img2.height = h;
    //    img2.Format() = ImageFormat(1,16,ImageElementFormat::UnsignedNormalized);
    SAIGA_ASSERT(0);
    img2.create();

    for(int i = 0; i < h; ++i)
    {
        for(int j = 0; j < w; ++j)
        {
#if 0
            float d = img.getPixel<float>(j,i);
            d = currentCamera->linearDepth(d);
            img2.getPixel<unsigned short>(j,i) = d * 0xFFFF;
#endif
        }
    }




    //    TextureLoader::instance()->saveImage(file,img2);
    img2.save(file);
#endif
}

std::string OpenGLWindow::getTimeString()
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


Ray OpenGLWindow::createPixelRay(const vec2 &pixel) const
{
    vec4 p = vec4(2*pixel.x/getWidth()-1.f,1.f-(2*pixel.y/getHeight()),0,1.f);
    p = glm::inverse(OpenGLWindow::currentCamera->proj)*p;
    p /= p.w;

    mat4 inverseView = glm::inverse(OpenGLWindow::currentCamera->view);
    vec3 ray_world =vec3(inverseView*p);
    vec3 origin = vec3(inverseView[3]);
    return Ray(glm::normalize(ray_world-origin),origin);
}

Ray OpenGLWindow::createPixelRay(const vec2 &pixel, const vec2& resolution, const mat4& inverseProj) const
{
    vec4 p = vec4(2*pixel.x/resolution.x-1.f,1.f-(2*pixel.y/resolution.y),0,1.f);
    p = inverseProj*p;
    p /= p.w;

    mat4 inverseView = glm::inverse(OpenGLWindow::currentCamera->view);
    vec3 ray_world =vec3(inverseView*p);
    vec3 origin = vec3(inverseView[3]);
    return Ray(glm::normalize(ray_world-origin),origin);
}

vec3 OpenGLWindow::screenToWorld(const vec2 &pixel) const
{
    vec4 p = vec4(2*pixel.x/getWidth()-1.f,1.f-(2*pixel.y/getHeight()),0,1.f);
    p = glm::inverse(OpenGLWindow::currentCamera->proj)*p;
    p /= p.w;

    mat4 inverseView = glm::inverse(OpenGLWindow::currentCamera->view);
    vec3 ray_world =vec3(inverseView*p);
    return ray_world;
}


vec3 OpenGLWindow::screenToWorld(const vec2 &pixel, const vec2& resolution, const mat4& inverseProj) const
{
    vec4 p = vec4(2*pixel.x/resolution.x-1.f,1.f-(2*pixel.y/resolution.y),0,1.f);
    p = inverseProj*p;
    p /= p.w;

    mat4 inverseView = glm::inverse(OpenGLWindow::currentCamera->view);
    vec3 ray_world =vec3(inverseView*p);
    return ray_world;
}



vec2 OpenGLWindow::projectToScreen(const vec3 &pos) const
{
    vec4 r = OpenGLWindow::currentCamera->proj * OpenGLWindow::currentCamera->view * vec4(pos,1);
    r /= r.w;

    vec2 pixel;
    pixel.x = (r.x +1.f)*getWidth() *0.5f;
    pixel.y = -(r.y - 1.f) * getHeight() * 0.5f;

    return pixel;
}

void OpenGLWindow::update(float dt)
{
    updateTimer.start();
    endParallelUpdate();
    if(updating)
        updating->update(dt);
    startParallelUpdate(dt);
    updateTimer.stop();
    updateUpdateGraph();

    numUpdates++;

    upsTimer.stop();
    upsTimer.start();
}




void OpenGLWindow::startParallelUpdate(float dt)
{

    if(parallelUpdate){
        semStartUpdate.notify();
    }else{
        parallelUpdateCaller(dt);
    }
}

void OpenGLWindow::endParallelUpdate()
{
    if(parallelUpdate)
        semFinishUpdate.wait();
}

void OpenGLWindow::parallelUpdateThread(float dt)
{
    semFinishUpdate.notify();
    semStartUpdate.wait();
    while(running){
        parallelUpdateCaller(dt);
        semFinishUpdate.notify();
        semStartUpdate.wait();
    }
}


void OpenGLWindow::parallelUpdateCaller(float dt)
{
    if(updating)
        updating->parallelUpdate(dt);
}

void OpenGLWindow::render(float dt, float interpolation)
{
    interpolationTimer.start();
    //    renderer->renderer->interpolate(dt,interpolation);
    if(updating)
        updating->interpolate(dt,interpolation);
    interpolationTimer.stop();

    renderCPUTimer.start();
    if(renderer)
        renderer->render_intern(currentCamera);
    renderCPUTimer.stop();

    updateRenderGraph();
    numFrames++;

    swapBuffersTimer.start();
    if(windowParameters.finishBeforeSwap) glFinish();
    swapBuffers();
    swapBuffersTimer.stop();

    fpsTimer.stop();
    fpsTimer.start();
}



void OpenGLWindow::sleep(tick_t ticks)
{
    if(ticks > tick_t(0)){
        std::this_thread::sleep_for(ticks);
    }
}




void OpenGLWindow::startMainLoop(MainLoopParameters params)
{
    parallelUpdate = params.parallelUpdate;
    printInfoMsg = params.printInfoMsg;
    gameTime.printInfoMsg = printInfoMsg;
    running = true;

    cout << "> Starting the main loop..." << endl;
    cout << "> updatesPerSecond=" << params.updatesPerSecond << " framesPerSecond=" << params.framesPerSecond << " maxFrameSkip=" << params.maxFrameSkip << endl;


    if(params.updatesPerSecond <= 0)
        params.updatesPerSecond = gameTime.base.count();
    if(params.framesPerSecond <= 0)
        params.framesPerSecond = gameTime.base.count();


    float updateDT = 1.0f / params.updatesPerSecond;
    targetUps = params.updatesPerSecond;
    //    float framesDT = 1.0f / framesPerSecond;

    tick_t ticksPerUpdate = gameTime.base / params.updatesPerSecond;
    tick_t ticksPerFrame = gameTime.base / params.framesPerSecond;

    tick_t ticksPerInfo = std::chrono::duration_cast<tick_t>(gameTime.base * params.mainLoopInfoTime);

    tick_t ticksPerScreenshot = std::chrono::duration_cast<tick_t>(gameTime.base * 5.0f);

    if(windowParameters.debugScreenshotTime < 0)
        ticksPerScreenshot = std::chrono::duration_cast<tick_t>(std::chrono::hours(100000));

    gameTime.init(ticksPerUpdate,ticksPerFrame);


    tick_t nextInfoTick = gameTime.getTime();
    tick_t nextScreenshotTick = gameTime.getTime() + ticksPerScreenshot;

    if(!params.catchUp){
        gameTime.maxGameLoopDelay = std::chrono::duration_cast<tick_t>(std::chrono::milliseconds(1));
    }


    if(parallelUpdate){
        updateThread = std::thread(&OpenGLWindow::parallelUpdateThread,this,updateDT);
    }


    while(true){
        checkEvents();

        if(shouldClose()){
            break;
        }

        //With this loop we are able to skip frames if the system can't keep up.
        for(int i = 0; i <= params.maxFrameSkip && gameTime.shouldUpdate(); ++i){
            update(updateDT);
        }

        if(gameTime.shouldRender()){
            render(updateDT,gameTime.interpolation);
        }

        if(printInfoMsg && gameTime.getTime() > nextInfoTick){
            auto gt = std::chrono::duration_cast<std::chrono::seconds>(gameTime.getTime());
            cout << "> Time: " << gt.count() << "s  Total number of updates/frames: " << numUpdates << "/" << numFrames << "  UPS/FPS: " << (1000.0f/upsTimer.getTimeMS()) << "/" << (1000.0f/fpsTimer.getTimeMS()) << endl;
            nextInfoTick += ticksPerInfo;
        }

        if(gameTime.getTime() > nextScreenshotTick){
            string file = windowParameters.debugScreenshotPath+getTimeString()+".png";
            this->screenshot(file);
            nextScreenshotTick += ticksPerScreenshot;
        }

        //sleep until the next interesting event
        sleep(gameTime.getSleepTime());
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

    auto gt = std::chrono::duration_cast<std::chrono::seconds>(gameTime.getTime());
    cout << "> Main loop finished in " << gt.count() << "s  Total number of updates/frames: " << numUpdates << "/" << numFrames  << endl;
}

}
