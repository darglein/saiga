/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <saiga/config.h>
#include <saiga/util/semaphore.h>
#include "saiga/time/gameTime.h"
#include "saiga/geometry/ray.h"
#include "saiga/imgui/imgui_renderer.h"

#include <thread>

namespace Saiga {

class Camera;
class Deferred_Renderer;
class Program;
struct RenderingParameters;
class Image;


struct SAIGA_GLOBAL OpenGLParameters{
    enum class Profile{
        ANY,
        CORE,
        COMPATIBILITY
    };
    Profile profile = Profile::CORE;

    bool debug = true;

    //all functionality deprecated in the requested version of OpenGL is removed
    bool forwardCompatible = false;

    int versionMajor = 3;
    int versionMinor = 2;

};

struct SAIGA_GLOBAL WindowParameters{
    enum class Mode{
        windowed,
        fullscreen,
        borderLessWindowed,
        borderLessFullscreen
    };

    std::string name = "Saiga";
    int width = 1280;
    int height = 720;
    Mode mode =  Mode::windowed;

    bool hidden = false; //for offscreen rendering
    bool alwaysOnTop = false;
    bool resizeAble = true;
    bool vsync = false;
    bool updateJoystick = false;
    int monitorId = 0; //Important for fullscreen mode. 0 is always the primary monitor.

    //time in seconds between debug screenshots. negativ for no debug screenshots
    float debugScreenshotTime = -1.0f;
    std::string debugScreenshotPath = "debug/";

    OpenGLParameters openglparameters;

    bool createImgui = true;
    std::string imguiFont = "fonts/SourceSansPro-Regular.ttf";
    int imguiFontSize = 15;

    bool borderLess(){ return mode==Mode::borderLessWindowed || mode==Mode::borderLessFullscreen;}
    bool fullscreen(){ return mode==Mode::fullscreen || mode==Mode::borderLessFullscreen;}

    void setMode(bool fullscreen, bool borderLess);
};

class SAIGA_GLOBAL OpenGLWindow{
protected:
    WindowParameters windowParameters;

    //total number of updateticks/frames rendered so far
    int numUpdates = 0;
    int numFrames = 0;

    //game loop running
    bool running = false;

    //basic variables for the parallel update
    Semaphore semStartUpdate, semFinishUpdate;
    std::thread updateThread;
    bool parallelUpdate = false;

    Deferred_Renderer* renderer = nullptr;
    Camera* currentCamera = nullptr;

    tick_t gameLoopDelay = tick_t(0);

    bool gameloopDropAccumulatedUpdates = false;
    bool printInfoMsg = true;

    //for imgui graph
    bool showImgui = true;
    static const int numGraphValues = 80;
    int imCurrentIndex = 0;
    float imUpdateTimes[numGraphValues] = {0};
    float imRenderTimes[numGraphValues] = {0};
    bool showImguiDemo = false;
public:
    bool showRendererImgui = false;
    std::shared_ptr<ImGuiRenderer> imgui;
    ExponentialTimer updateTimer, interpolationTimer, renderCPUTimer, swapBuffersTimer;
    AverageTimer fpsTimer, upsTimer;
public:
    OpenGLWindow(WindowParameters windowParameters);
    virtual ~OpenGLWindow();

    void setProgram(Program* program);
    bool init(const RenderingParameters &params);
    void startMainLoop(int updatesPerSecond, int framesPerSecond, float mainLoopInfoTime=5.0f, int maxFrameSkip = 0, bool _parallelUpdate=false, bool _catchUp=false, bool _printInfoMsg=true);
    void close();
    void renderImGui(bool* p_open = NULL);


    //uses the current camera to project between world and screen
    Ray createPixelRay(const vec2 &pixel) const;
    Ray createPixelRay(const vec2 &pixel, const vec2 &resolution, const mat4 &inverseProj) const;
    vec2 projectToScreen(const vec3 &pos) const;
    vec3 screenToWorld(const vec2 &pixel) const;
    vec3 screenToWorld(const vec2 &pixel, const vec2& resolution, const mat4& inverseProj) const;

    //reading the default framebuffer
    void readToExistingImage(Image &out);
    void readToImage(Image &out);

    //read the default framebuffer and save to file
    void screenshot(const std::string &file);
    void screenshotRender(const std::string &file);


    //Basic getters and setters

    std::string getTimeString();
    int getWidth() const { return windowParameters.width; }
    int getHeight() const { return windowParameters.height; }
    float getAspectRatio() const { return (float)windowParameters.width/(float)windowParameters.height; }
    Camera* getCamera() const { return currentCamera; }
    std::string getName() const { return windowParameters.name; }
    void setCamera(Camera* c) { currentCamera = c; }
    Deferred_Renderer* getRenderer() const {  return renderer; }


protected:
    void resize(int width, int height);
    void initDeferredRendering(const RenderingParameters& params);
    void update(float dt);
    void render(float dt, float interpolation);
    void startParallelUpdate(float dt);
    void parallelUpdateCaller(float dt);
    void endParallelUpdate();
    void parallelUpdateThread(float dt);


    virtual bool initWindow() = 0;
    virtual bool initInput() = 0;
    virtual bool shouldClose() { return !running; }
    virtual void checkEvents() = 0;
    virtual void swapBuffers() = 0;
    virtual void freeContext() = 0;

    void sleep(tick_t ticks);

};

}
