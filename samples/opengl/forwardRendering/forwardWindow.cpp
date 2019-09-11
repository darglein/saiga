/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "forwardWindow.h"

#include "saiga/core/geometry/triangle_mesh_generator.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/math/random.h"
#include "saiga/opengl/shader/shaderLoader.h"


Sample::Sample(OpenGLWindow& window, OpenGLRenderer& renderer) : Updating(window), ForwardRenderingInterface(renderer)
{
    // create a perspective camera
    float aspect = window.getAspectRatio();
    camera.setProj(60.0f, aspect, 0.1f, 50.0f);
    camera.setView(vec3(0, 5, 10), vec3(0, 0, 0), vec3(0, 1, 0));

    // Set the camera from which view the scene is rendered
    window.setCamera(&camera);


    // add this object to the keylistener, so keyPressed and keyReleased will be called


    // This simple AssetLoader can create assets from meshes and generate some generic debug assets
    AssetLoader assetLoader;
    groundPlane.asset = assetLoader.loadDebugGrid(20, 20, 1);



    for (int i = 0; i < 10000; ++i)
    {
        PointVertex v;
        v.position = linearRand(make_vec3(-3), make_vec3(3));
        v.color    = linearRand(make_vec3(0), make_vec3(1));
        pointCloud.points.push_back(v);
    }
    pointCloud.updateBuffer();

    {
        // let's just draw the coordiante axis
        PointVertex v;

        // x
        v.color    = vec3(1, 0, 0);
        v.position = vec3(-1, 0, 0);
        lineSoup.lines.push_back(v);
        v.position = vec3(1, 0, 0);
        lineSoup.lines.push_back(v);

        // y
        v.color    = vec3(0, 1, 0);
        v.position = vec3(0, -1, 0);
        lineSoup.lines.push_back(v);
        v.position = vec3(0, 1, 0);
        lineSoup.lines.push_back(v);

        // z
        v.color    = vec3(0, 0, 1);
        v.position = vec3(0, 0, -1);
        lineSoup.lines.push_back(v);
        v.position = vec3(0, 0, 1);
        lineSoup.lines.push_back(v);

        lineSoup.translateGlobal(vec3(5, 5, 5));
        lineSoup.calculateModel();

        lineSoup.lineWidth = 3;
        lineSoup.updateBuffer();
    }


    //    frustum.vertices.resize(2);
    //    frustum.vertices[0].position = vec4(0, 0, 0, 0);
    //    frustum.vertices[1].position = vec4(10, 10, 10, 0);
    //    frustum.fromLineList();

    //    frustum.createGrid({100, 100}, {0.1, 0.1});
    frustum.createFrustum(camera.proj, 1);
    frustum.setColor(vec4{0, 1, 0, 1});

    auto shader = ShaderLoader::instance()->load<MVPShader>("colored_points.glsl");
    frustum.create(shader, shader, shader, shader);
    frustum.loadDefaultShaders();
    //    frustum.model.createFrustum(camera.proj);
    //    assetLoader.frustumMesh();


    std::cout << "Program Initialized!" << std::endl;
}

Sample::~Sample() {}

void Sample::update(float dt)
{
    // Update the camera position
    camera.update(dt);
}

void Sample::interpolate(float dt, float interpolation)
{
    // Update the camera rotation. This could also be done in 'update' but
    // doing it in the interpolate step will reduce latency
    camera.interpolate(dt, interpolation);
}



void Sample::renderOverlay(Camera* cam)
{
    // The skybox is rendered after lighting and before post processing
    skybox.render(cam);

    // Render all objects from the viewpoint of 'cam'
    groundPlane.renderForward(cam);

    pointCloud.render(cam);

    lineSoup.render(cam);

    frustum.renderForward(cam, mat4::Identity());
}



void Sample::renderFinal(Camera* cam)
{
    // The final render path (after post processing).
    // Usually the GUI is rendered here.

    parentWindow.renderImGui();

    {
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(400, 200), ImGuiCond_FirstUseEver);
        ImGui::Begin("An Imgui Window :D");

        ImGui::End();
    }

    {
        // console like output
        oc.render();
    }
}


void Sample::keyPressed(SDL_Keysym key)
{
    switch (key.scancode)
    {
        case SDL_SCANCODE_ESCAPE:
            parentWindow.close();
            break;
        default:
            break;
    }
}

void Sample::keyReleased(SDL_Keysym key) {}
