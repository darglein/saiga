/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "advancedWindow.h"

#include "saiga/core/geometry/triangle_mesh_generator.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/opengl/shader/shaderLoader.h"

Sample::Sample(OpenGLWindow& window, Renderer& renderer)
    : Updating(window), ForwardRenderingInterface(renderer), tdo(window.getWidth(), window.getHeight())
{
    window.setCamera(&camera);

    textAtlas.loadFont("SourceSansPro-Regular.ttf", 40, 2, 4, true);

    tdo.init(&textAtlas);
    tdo.borderX  = 0.01f;
    tdo.borderY  = 0.01f;
    tdo.paddingY = 0.000f;
    tdo.textSize = 0.8f;

    tdo.textParameters.setColor(vec4(1), 0.1f);
    tdo.textParameters.setGlow(vec4(0, 0, 0, 1), 1.0f);

    tdo.createItem("Time (ms): ");


    timer.start();
    std::cout << "Program Initialized!" << std::endl;
}

Sample::~Sample()
{
    // We don't need to delete anything here, because objects obtained from saiga are wrapped in smart pointers.
}

void Sample::update(float dt) {}

void Sample::interpolate(float dt, float interpolation)
{
    if (running)
    {
        auto time                                     = timer.stop();
        std::chrono::duration<double, std::milli> tms = time;
        tdo.updateEntry(0, tms.count());
    }
}



void Sample::renderOverlay(Camera* cam)
{
    // The skybox is rendered after lighting and before post processing
    //    skybox.render(cam);
}

void Sample::renderFinal(Camera* cam)
{
    // The final render path (after post processing).
    // Usually the GUI is rendered here.

    tdo.layout.cam.calculateModel();
    tdo.layout.cam.recalculateMatrices();
    tdo.layout.cam.recalculatePlanes();

    parentWindow.getRenderer()->bindCamera(&tdo.layout.cam);
    tdo.render();
}


void Sample::keyPressed(SDL_Keysym key)
{
    switch (key.scancode)
    {
        case SDL_SCANCODE_ESCAPE:
            parentWindow.close();
            break;
        case SDL_SCANCODE_R:
            timer.start();
            break;
        case SDL_SCANCODE_SPACE:
            running = !running;
            break;
        default:
            break;
    }
}

void Sample::keyReleased(SDL_Keysym key) {}
