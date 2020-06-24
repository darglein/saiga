/**
 * Copyright (c) 2020 Simon Mederer
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/sdl/sdl_camera.h"
#include "saiga/core/sdl/sdl_eventhandler.h"
#include "saiga/core/util/statistics.h"
#include "saiga/opengl/assets/all.h"
#include "saiga/opengl/assets/objAssetLoader.h"
#include "saiga/opengl/rendering/deferredRendering/deferredRendering.h"
#include "saiga/opengl/rendering/renderer.h"
#include "saiga/opengl/window/SampleWindowDeferred.h"
#include "saiga/opengl/window/SampleWindowForward.h"
#include "saiga/opengl/window/sdl_window.h"
#include "saiga/opengl/world/proceduralSkybox.h"
#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/util/DepthmapPreprocessor.h"

#include "decimate.h"

#include <set>

#include "image_triangulator.h"

using namespace Saiga;

class Sample : public StandaloneWindow<WindowManagement::SDL, Forward_Renderer>, public SDL_KeyListener
{
   public:
    DepthProcessor2::Settings ip_settings;
    SimpleTriangulator::Settings st_settings;
    RQT_Triangulator::Settings rqt_settings;
    QuadricDecimater::Settings qd_settings;

    bool wireframe = true;

    char depth_image_input[100] = "bar.saigai";

    // camera parameters used for the bar.saigai image
    StereoCamera4Base<float> cameraParameters = StereoCamera4Base<float>(
        5.3887405952849110e+02, 5.3937051275591125e+02, 3.2233507920081263e+02, 2.3691517848391885e+02, 40.0f);

    TemplatedImage<float> loaded_depth_image;

    SimpleAssetObject meshObject;

    TriangleMesh<VertexNC, GLuint> depthmesh;

    Sample();

    // loads a depth image named by depth_image_input
    void load_depth_image();

    // uses depthmapPreprocessor to scale down the loaded depth map
    void scale_down_depth_image();

    // preprocesses the depth image given by the ImGui window using different methods and writes it as file with the
    // second name given in the ImGui window
    void preprocess_occlusion_edges();

    // uses a gaussian filter to blur the loaded depth map
    void blur_depth_image();

    // takes loaded_depth_image and triangulates all pixels. The result is written into baseMesh
    void triangulate_naive();

    // takes loaded_depth_image and triangulates the Pixels using a restricted quadtree as shown in
    // "Large Scale Terrain Visualization Using The Restricted Quadtree Triangulation"
    void triangulate_RQT();

    // reduces the current baseMesh using the quadric Error metric
    void reduce_quadric();

    void update(float dt) override;
    void interpolate(float dt, float interpolation) override;

    void renderOverlay(Camera* cam) override;
    void renderFinal(Camera* cam) override;

    void keyPressed(SDL_Keysym key) override;
    void keyReleased(SDL_Keysym key) override;

    SDLCamera<PerspectiveCamera> camera;
};
