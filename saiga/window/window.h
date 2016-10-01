#pragma once

#include "saiga/util/timer2.h"
#include "saiga/geometry/ray.h"
#include <saiga/config.h>
#include <string>
#include <mutex>
#include <memory>
#include <list>
#include <thread>

class Camera;
class Deferred_Renderer;
class Program;
struct RenderingParameters;
class Image;

typedef long long tick_t;

class SAIGA_GLOBAL Window{
public:



    std::string name;
    int width;
    int height;

    bool running = true;

    Deferred_Renderer* renderer = nullptr;
    Camera* currentCamera = nullptr;

    ExponentialTimer updateTimer, interpolationTimer, renderCPUTimer;
    AverageTimer fpsTimer, upsTimer;


    virtual bool initWindow() = 0;
    virtual bool initInput() = 0;
public:

    Window(const std::string &name,int width,int height);
     virtual ~Window();


    void quit();
    bool init(const RenderingParameters &params);
    void initDeferredRendering(const RenderingParameters& params);

    virtual void close() = 0;

    void resize(int width, int height);

    void screenshot(const std::string &file);
    void screenshotRender(const std::string &file);
    std::string getTimeString();

    void setVideoRecordingLimit(int limit){queueLimit = limit;}


    void setProgram(Program* program);

    int getWidth();
    int getHeight();
    float getAspectRatio();
    Camera* getCamera();
    void setCamera(Camera* c);
    Deferred_Renderer* getRenderer();


    Ray createPixelRay(const glm::vec2 &pixel) const;
    Ray createPixelRay(const glm::vec2 &pixel, const glm::vec2 &resolution, const glm::mat4 &inverseProj) const;
    vec2 projectToScreen(const glm::vec3 &pos) const;
    void screenshotParallelWrite(const std::string &file);
    vec3 screenToWorld(const glm::vec2 &pixel) const;
    vec3 screenToWorld(const glm::vec2 &pixel, const vec2& resolution, const mat4& inverseProj) const;

    void startMainLoop(int updatesPerSecond, int framesPerSecond);
    virtual bool shouldClose() { return !running; }
    virtual void swapBuffers() = 0;
    virtual void checkEvents() = 0;
protected:
    void update(float dt);
    void render(float dt, float interpolation);


    Timer2 gameTimer;
    //the game ticks are the microseconds since the start
    tick_t getGameTicks();
    tick_t getGameTicksPerSecond() { return 1000000; }
    void sleep(tick_t ticks);

private:
    int currentScreenshot = 0;
    std::string parallelScreenshotPath;

    std::list<std::shared_ptr<Image>> queue;
    std::mutex lock;
    bool ssRunning = false;
    int queueLimit = 200;

    bool waitForWriters = false;

#define WRITER_COUNT 7
    std::thread* sswriterthreads[WRITER_COUNT];
    void processScreenshots();
};

inline int Window::getWidth(){
    return width;
}

inline int Window::getHeight(){
    return height;
}

inline float Window::getAspectRatio(){
    return (float)width/(float)height;
}

inline Camera *Window::getCamera(){
    return currentCamera;
}

inline void Window::setCamera(Camera *c){
    currentCamera = c;
}

inline Deferred_Renderer *Window::getRenderer(){
    return renderer;
}
