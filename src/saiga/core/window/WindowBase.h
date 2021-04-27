/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/camera/camera.h"
#include "saiga/core/geometry/ray.h"

#include "Interfaces.h"
#include "MainLoop.h"
#include "windowParameters.h"
namespace Saiga
{
class SAIGA_CORE_API WindowBase : public MainLoopInterface
{
   public:
    MainLoop mainLoop;

    WindowBase(WindowParameters windowParameters);
    virtual ~WindowBase();
    void close();

    void startMainLoop(MainLoopParameters params = MainLoopParameters());

    virtual void renderImGui(bool* p_open = nullptr) = 0;


    std::string getTimeString();
    int getWidth() const { return windowParameters.width; }
    int getHeight() const { return windowParameters.height; }
    float getAspectRatio() const { return (float)windowParameters.width / (float)windowParameters.height; }
    std::string getName() const { return windowParameters.name; }

    // these functions are only valid in single camera mode
    Camera* getCamera() const { return activeCameras.empty() ? nullptr : activeCameras.front().first; }
    void setCamera(Camera* c)
    {
        activeCameras.resize(1);
        activeCameras.front() = {c, ViewPort({0, 0}, {getWidth(), getHeight()})};
    }

    // set up multi camera
    void setMultiCamera(const std::vector<std::pair<Camera*, ViewPort>>& cameras) { activeCameras = cameras; }

    RendererBase* getRenderer() const { return renderer; }

    void setUpdateObject(Updating& u) { updating = &u; }
    void setRenderer(RendererBase* u) { renderer = u; }
    void setShowImgui(bool b) { showImgui = b; }

    virtual bool shouldClose() { return !running; }
    virtual void render();
    virtual void interpolate(float dt, float alpha);

   protected:
    void resize(int width, int height);

    bool running = false;
    WindowParameters windowParameters;

    // all these cameras will be rendered
    std::vector<std::pair<Camera*, ViewPort>> activeCameras;

    //    Camera* currentCamera  = nullptr;
    bool showImgui         = false;
    RendererBase* renderer = nullptr;
    Updating* updating     = nullptr;
};

}  // namespace Saiga
