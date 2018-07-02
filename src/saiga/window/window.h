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
#include "saiga/window/windowParameters.h"
#include <thread>
#include "saiga/window/MainLoop.h"

namespace Saiga {

class Camera;
class Deferred_Renderer;
struct DeferredRenderingParameters;
class Image;

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


    bool init(const DeferredRenderingParameters &params);
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
