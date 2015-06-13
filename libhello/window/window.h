#pragma once

#include <string>

#include "libhello/opengl/objloader.h"

class Camera;
class Deferred_Renderer;

class Window{
protected:
    std::string name;
    int window_width;
    int window_height;
    bool fullscreen = false;

    bool running;

    Deferred_Renderer* renderer;
    Camera* currentCamera = nullptr;
    ObjLoader objLoader;
    TextureLoader textureLoader;
    MaterialLoader materialLoader;
    ShaderLoader shaderLoader;

    virtual bool initWindow() = 0;
    virtual bool initInput() = 0;
    virtual void update(float delta) = 0;
public:
    bool vsync = false;

    Window(const std::string &name,int window_width,int window_height, bool fullscreen);
     virtual ~Window(){}

    void quit(){cout<<"Window: Quit"<<endl;running = false;}
    bool init();
    virtual void close() = 0;

    void screenshot(const string &file);
    string getTimeString();

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

    template<typename shader_t>
    inline shader_t* loadShader(const std::string &name){
        return shaderLoader.load<shader_t>(name);
    }

    Texture* loadTexture(const std::string &name){
        return textureLoader.load(name);
    }

    material_mesh_t* loadObj(const std::string &name){
        return objLoader.load(name);
    }

};
