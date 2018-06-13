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
#include "saiga/rendering/renderer.h"

#include <thread>

namespace Saiga {

class Camera;
class Deferred_Renderer;
struct RenderingParameters;
class Image;


struct SAIGA_GLOBAL OpenGLParameters
{
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

struct SAIGA_GLOBAL WindowParameters
{
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

    bool finishBeforeSwap = false; //adds a glFinish before swapBuffers
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
    std::string imguiFont = "";
    int imguiFontSize = 15;

    bool borderLess(){ return mode==Mode::borderLessWindowed || mode==Mode::borderLessFullscreen;}
    bool fullscreen(){ return mode==Mode::fullscreen || mode==Mode::borderLessFullscreen;}

    void setMode(bool fullscreen, bool borderLess);
};


struct SAIGA_GLOBAL MainLoopParameters
{

    /**
     * @brief startMainLoop
     * @param updatesPerSecond
     *      Number of calls per second to the virtual function "update".
     *      A value of 0 means: update as fast as possible (not recommended)
     * @param framesPerSecond
     *      Number of class per second to the render functions.
     *      A value of 0 is unlimitted frames.
     * @param mainLoopInfoTime
     *      Time between mainloop debug output to the console
     * @param maxFrameSkip
     *      Max number of frames that are skipped if the update cannot keep up.
     * @param _parallelUpdate
     *      Enables parallel updates while rendering. This will call the virtual function parallelUpdate.
     * @param _catchUp
     *      Lets the update loop catch up in case of lags.
     * @param _printInfoMsg
     *      Enable/Disable the debug output
     */
    int updatesPerSecond = 60;
    int framesPerSecond = 60;
    float mainLoopInfoTime=5.0f;
    int maxFrameSkip = 0;
    bool _parallelUpdate=false;
    bool catchUp=false;
    bool _printInfoMsg=true;
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



    Camera* currentCamera = nullptr;

    tick_t gameLoopDelay = tick_t(0);

    bool gameloopDropAccumulatedUpdates = false;
    bool printInfoMsg = true;

    //for imgui graph
    bool showImgui = true;
    static const int numGraphValues = 80;
    float ut=0, ft=0;
    float avFt = 0, avUt;
    int imCurrentIndexUpdate = 0;
    int imCurrentIndexRender = 0;
    float imUpdateTimes[numGraphValues];
    float imRenderTimes[numGraphValues];
    bool showImguiDemo = false;
    float maxUpdateTime = 1;
    float maxRenderTime = 1;
    int targetUps = 60;
    Renderer* renderer = nullptr;
    Updating* updating = nullptr;
public:

    bool showRendererImgui = false;

    ExponentialTimer updateTimer, interpolationTimer, renderCPUTimer, swapBuffersTimer;
    AverageTimer fpsTimer, upsTimer;
public:
    OpenGLWindow(WindowParameters windowParameters);
    virtual ~OpenGLWindow();


    bool init(const RenderingParameters &params);
    bool create();
    void destroy();


    void startMainLoop(MainLoopParameters params = MainLoopParameters());


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
    void screenshotRenderDepth(const std::string &file);
    void getDepthFloat(Image &out);

    //Basic getters and setters

    std::string getTimeString();
    int getWidth() const { return windowParameters.width; }
    int getHeight() const { return windowParameters.height; }
    float getAspectRatio() const { return (float)windowParameters.width/(float)windowParameters.height; }
    Camera* getCamera() const { return currentCamera; }
    std::string getName() const { return windowParameters.name; }
    void setCamera(Camera* c) { currentCamera = c; }
    Renderer* getRenderer() const {  return renderer; }
    int getTargetUps() const { return targetUps; }

    void setUpdateObject(Updating &u) { updating = &u; }
    void setRenderer(Renderer *u) { renderer = u; }

    void setShowImgui(bool b) { showImgui = b; }


    virtual std::shared_ptr<ImGuiRenderer> createImGui() { return nullptr; }
protected:
    void resize(int width, int height);
    void initDeferredRendering(const RenderingParameters& params);
    void update(float dt);
    void render(float dt, float interpolation);
    void startParallelUpdate(float dt);
    void parallelUpdateCaller(float dt);
    void endParallelUpdate();
    void parallelUpdateThread(float dt);

    void updateRenderGraph();
    void updateUpdateGraph();

    virtual bool initWindow() = 0;
    virtual bool initInput() = 0;
    virtual bool shouldClose() { return !running; }
    virtual void checkEvents() = 0;
    virtual void swapBuffers() = 0;
    virtual void freeContext() = 0;

    void sleep(tick_t ticks);

};

}
