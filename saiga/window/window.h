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
class fipImage;

class SAIGA_GLOBAL Window{
public:
    std::string name;
    int width;
    int height;

    bool running;

    Deferred_Renderer* renderer = nullptr;
    Camera* currentCamera = nullptr;

    ExponentialTimer updateTimer, interpolationTimer, renderCPUTimer;
    AverageTimer fpsTimer;


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
    std::string getTimeString();


    void setProgram(Program* program);

    int getWidth();
    int getHeight();
    float getAspectRatio();
    Camera* getCamera();
    void setCamera(Camera* c);
    Deferred_Renderer* getRenderer();


    Ray createPixelRay(const glm::vec2 &pixel);
    vec2 projectToScreen(const glm::vec3 &pos);
    void screenshotParallelWrite(const std::string &file);
protected:
    void update(float dt);
    void render(float interpolation = 0.0f);

private:
    int currentScreenshot = 0;
    std::string parallelScreenshotPath;

    std::list<std::shared_ptr<fipImage>> queue;
    std::mutex lock;
    bool ssRunning = false;

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
