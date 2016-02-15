#pragma once

#include "saiga/geometry/ray.h"
#include <saiga/config.h>
#include <string>

class Camera;
class Deferred_Renderer;
class Program;
class RenderingParameters;

class SAIGA_GLOBAL Window{
public:
    std::string name;
    int width;
    int height;
    bool fullscreen = false;

    bool running;

    Deferred_Renderer* renderer;
    Camera* currentCamera = nullptr;


    virtual bool initWindow() = 0;
    virtual bool initInput() = 0;
    virtual void update(float delta) {}
public:
    bool vsync = false;

    Window(const std::string &name,int width,int height, bool fullscreen);
     virtual ~Window(){}


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
