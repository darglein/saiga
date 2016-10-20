#pragma once

#include <saiga/config.h>
#include <thread>
#include <saiga/util/semaphore.h>
#include "saiga/util/timer2.h"
#include "saiga/geometry/ray.h"



class Camera;
class Deferred_Renderer;
class Program;
struct RenderingParameters;
class Image;



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

    bool alwaysOnTop = false;
    bool resizeAble = true;
    bool vsync = false;
    bool updateJoystick = false;
    bool debugContext = true;
    bool coreContext = true;
    int monitorId = 0; //Important for fullscreen mode. 0 is always the primary monitor.

    bool borderLess(){ return mode==Mode::borderLessWindowed || mode==Mode::borderLessFullscreen;}
    bool fullscreen(){ return mode==Mode::fullscreen || mode==Mode::borderLessFullscreen;}

    void setMode(bool fullscreen, bool borderLess);
};

class SAIGA_GLOBAL Window{
protected:
    WindowParameters windowParameters;
    int numUpdates = 0;
    int numFrames = 0;

    Timer2 gameTimer;
    double timeScale = 1.f;
    bool running = true;

    Semaphore semStartUpdate, semFinishUpdate;
    std::thread updateThread;
    bool parallelUpdate = false;

    Deferred_Renderer* renderer = nullptr;
    Camera* currentCamera = nullptr;

    bool gameloopDropAccumulatedUpdates = false;
public:
    ExponentialTimer updateTimer, interpolationTimer, renderCPUTimer, swapBuffersTimer;
    AverageTimer fpsTimer, upsTimer;
public:
    Window(WindowParameters windowParameters);
    virtual ~Window();

    void setProgram(Program* program);
    bool init(const RenderingParameters &params);
    void startMainLoop(int updatesPerSecond, int framesPerSecond, float mainLoopInfoTime=5.0f, int maxFrameSkip = 0, bool _parallelUpdate=false, bool _catchUp=false);
    void close();
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

public:

    //Basic getters and setters

    std::string getTimeString();
    int getWidth() const { return windowParameters.width; }
    int getHeight() const { return windowParameters.height; }
    float getAspectRatio() const { return (float)windowParameters.width/(float)windowParameters.height; }
    Camera* getCamera() const { return currentCamera; }
    std::string getName() const { return windowParameters.name; }
    void setTimeScale(double timeScale);
    void setCamera(Camera* c) { currentCamera = c; }
    Deferred_Renderer* getRenderer() const {  return renderer; }

    //When there are very long updates for example when loading a level, the
    //gameloop tries to catch up, unless you call this method.
    void jumpGameLoopToLive(){ gameloopDropAccumulatedUpdates = true; }

    //number of ticks since startgameloop has been called
    tick_t getGameTicks();

    void screenshot(const std::string &file);
    void screenshotRender(const std::string &file);


    Ray createPixelRay(const vec2 &pixel) const;
    Ray createPixelRay(const vec2 &pixel, const vec2 &resolution, const mat4 &inverseProj) const;
    vec2 projectToScreen(const vec3 &pos) const;
    vec3 screenToWorld(const vec2 &pixel) const;
    vec3 screenToWorld(const vec2 &pixel, const vec2& resolution, const mat4& inverseProj) const;
};




