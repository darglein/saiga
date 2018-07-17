/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/rendering/renderer.h"

#include "saiga/camera/camera.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/OpenGLWindow.h"

namespace Saiga {

Renderer::Renderer(OpenGLWindow &window)
    : outputWidth(window.getWidth()), outputHeight(window.getHeight())
{
    cameraBuffer.createGLBuffer(nullptr,sizeof(CameraDataGLSL),GL_DYNAMIC_DRAW);


    window.setRenderer(this);

    // ImGUI
    imgui = window.createImGui();
}

Renderer::~Renderer()
{

}

void Renderer::resize(int windowWidth, int windowHeight)
{
    outputWidth = windowWidth;
    outputHeight = windowHeight;
//    cout << "resize to " << windowWidth << "x" << windowHeight << endl;
}


void Renderer::bindCamera(Camera *cam)
{
    CameraDataGLSL cd(cam);
    cameraBuffer.updateBuffer(&cd,sizeof(CameraDataGLSL),0);
    cameraBuffer.bind(CAMERA_DATA_BINDING_POINT);
}


}
