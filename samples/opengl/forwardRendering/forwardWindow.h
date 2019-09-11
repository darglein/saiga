/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/sdl/all.h"
#include "saiga/opengl/assets/all.h"
#include "saiga/opengl/assets/objAssetLoader.h"
#include "saiga/opengl/rendering/forwardRendering/forwardRendering.h"
#include "saiga/opengl/rendering/renderer.h"
#include "saiga/opengl/window/sdl_window.h"
#include "saiga/opengl/world/LineSoup.h"
#include "saiga/opengl/world/pointCloud.h"
#include "saiga/opengl/world/proceduralSkybox.h"

#include <sstream>
using namespace Saiga;


struct OutputConsole
{
    void render()
    {
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(400, 200), ImGuiCond_FirstUseEver);
        ImGui::Begin(name.c_str());



        ImGui::End();
    }

    template <typename T>
    OutputConsole& operator<<(const T& t)
    {
        std::stringstream strm;
        strm << t;
        content += strm.str();

        return *this;
    }

    std::string name = "Output";
    std::string content;
};

class Sample : public Updating, public ForwardRenderingInterface, public SDL_KeyListener
{
   public:
    SDLCamera<PerspectiveCamera> camera;

    GLPointCloud pointCloud;
    LineSoup lineSoup;
    SimpleAssetObject groundPlane;
    LineVertexColoredAsset frustum;


    OutputConsole oc;
    ProceduralSkybox skybox;

    std::shared_ptr<Texture> t;

    Sample(OpenGLWindow& window, OpenGLRenderer& renderer);
    ~Sample();

    void update(float dt) override;
    void interpolate(float dt, float interpolation) override;
    void renderOverlay(Camera* cam) override;
    void renderFinal(Camera* cam) override;

    void keyPressed(SDL_Keysym key) override;
    void keyReleased(SDL_Keysym key) override;
};
