/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <saiga/config.h>
#include "saiga/geometry/ray.h"
#include "saiga/window/Interfaces.h"
#include "saiga/rendering/renderer.h"
#include "saiga/window/windowParameters.h"
#include "saiga/window/MainLoop.h"

namespace Saiga {

class Camera;
class Deferred_Renderer;
struct DeferredRenderingParameters;
class Image;

class SAIGA_GLOBAL WindowBase : public MainLoopInterface
{
protected:
    bool running = false;
    WindowParameters windowParameters;

    MainLoop mainLoop;


    Camera* currentCamera = nullptr;


    //for imgui graph
    bool showImgui = true;

    Renderer* renderer = nullptr;
    Updating* updating = nullptr;
public:

    bool showRendererImgui = false;


public:
    WindowBase(WindowParameters windowParameters);
    virtual ~WindowBase();


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


    std::string getTimeString();
    int getWidth() const { return windowParameters.width; }
    int getHeight() const { return windowParameters.height; }
    float getAspectRatio() const { return (float)windowParameters.width/(float)windowParameters.height; }
    Camera* getCamera() const { return currentCamera; }
    std::string getName() const { return windowParameters.name; }
    void setCamera(Camera* c) { currentCamera = c; }
    Renderer* getRenderer() const {  return renderer; }


    void setUpdateObject(Updating &u) { updating = &u; }
    void setRenderer(Renderer *u) { renderer = u; }

    void setShowImgui(bool b) { showImgui = b; }


    virtual void render();
    virtual void swap(){}
    virtual float getTotalRenderTime() { return 0; }
//    virtual bool shouldClose() { return false; }
    virtual void update(float dt){}
    virtual void parallelUpdate(float dt) { (void)dt; }
    virtual void interpolate(float dt, float alpha);


protected:
    void resize(int width, int height);


};

}
