#pragma once

#include <string>

class Camera;
class Deferred_Renderer;

class Window{
public:
    std::string name;
    int window_width;
    int window_height;
    bool fullscreen = false;

    bool running;

    Deferred_Renderer* renderer;
    Camera* currentCamera = nullptr;
//    ObjLoader objLoader;
//    TextureLoader textureLoader;
//    MaterialLoader materialLoader;
//    ShaderLoader shaderLoader;

    virtual bool initWindow() = 0;
    virtual bool initInput() = 0;
    virtual void update(float delta) = 0;
public:
    bool vsync = false;

    Window(const std::string &name,int window_width,int window_height, bool fullscreen);
     virtual ~Window(){}

    void quit();
    bool init();
    virtual void close() = 0;

    void screenshot(const std::string &file);
    std::string getTimeString();

    inline int getWidth(){
        return window_width;
    }

    inline int getHeight(){
        return window_height;
    }

    inline float getAspectRatio(){
        return (float)window_width/(float)window_height;
    }


    inline Camera* getCamera(){
        return currentCamera;
    }

    inline void setCamera(Camera* c){
        currentCamera = c;
    }

    inline Deferred_Renderer* getRenderer(){
        return renderer;
    }


};
