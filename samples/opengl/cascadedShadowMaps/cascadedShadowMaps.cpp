/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "cascadedShadowMaps.h"

#include "saiga/core/geometry/triangle_mesh_generator.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/math/random.h"
#include "saiga/opengl/shader/shaderLoader.h"

Sample::Sample()
{
    // This simple AssetLoader can create assets from meshes and generate some generic debug assets
    AssetLoader assetLoader;

    float height = 20;
    // First create the triangle mesh of a cube
    auto cubeMesh = TriangleMeshGenerator::createMesh(AABB(vec3(-1, 0, -1), vec3(1, height, 1)));

    // To render a triangle mesh we need to wrap it into an asset. This creates the required OpenGL buffers and provides
    // render functions.
    auto cubeAsset = assetLoader.assetFromMesh(*cubeMesh, Colors::blue);

    float s = 200;

    for (int i = 0; i < 500; ++i)
    {
        SimpleAssetObject cube;
        cube.asset = cubeAsset;
        cube.translateGlobal(linearRand(vec3(-s, 0, -s), vec3(s, 0, s)));
        cube.calculateModel();
        cubes.push_back(cube);
    }

    sceneBB = AABB(vec3(-s, 0, -s), vec3(s, height, s));

    groundPlane.asset = assetLoader.loadDebugPlaneAsset(vec2(s, s), 1.0f, Colors::lightgray, Colors::gray);

    // create one directional light

    Base::sun->setActive(false);

    sun = renderer->lighting.createDirectionalLight();
    sun->setDirection(vec3(-1, -2, -2.5));
    //    sun->setDirection(vec3(0,-1,0));
    sun->setColorDiffuse(LightColorPresets::DirectSunlight);
    sun->setIntensity(1.0);
    sun->setAmbientIntensity(0.02f);
    sun->createShadowMap(512, 512);
    sun->enableShadows();

    camera.recalculateMatrices();
    camera.recalculatePlanes();

    sun->fitShadowToCamera(&camera);
    sun->fitNearPlaneToScene(sceneBB);

    std::cout << "Program Initialized!" << std::endl;
}

Sample::~Sample()
{
    // We don't need to delete anything here, because objects obtained from saiga are wrapped in smart pointers.
}

void Sample::update(float dt)
{
    // Update the camera position
    camera.update(dt);
    if (fitShadowToCamera)
    {
        sun->fitShadowToCamera(&camera);
    }
    if (fitNearPlaneToScene)
    {
        sun->fitNearPlaneToScene(sceneBB);
    }
}

void Sample::interpolate(float dt, float interpolation)
{
    // Update the camera rotation. This could also be done in 'update' but
    // doing it in the interpolate step will reduce latency
    camera.interpolate(dt, interpolation);
}

void Sample::render(Camera* cam)
{
    // Render all objects from the viewpoint of 'cam'
    groundPlane.render(cam);

    for (auto& c : cubes)
    {
        c.render(cam);
    }
}

void Sample::renderDepth(Camera* cam)
{
    // Render the depth of all objects from the viewpoint of 'cam'
    // This will be called automatically for shadow casting light sources to create shadow maps
    groundPlane.renderDepth(cam);

    for (auto& c : cubes)
    {
        c.renderDepth(cam);
    }
}

void Sample::renderOverlay(Camera* cam)
{
    // The skybox is rendered after lighting and before post processing
    skybox.render(cam);
}

void Sample::renderFinal(Camera* cam)
{
    {
        ImGui::SetNextWindowPos(ImVec2(50, 50), ImGuiCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(400, 200), ImGuiCond_FirstUseEver);
        ImGui::Begin("An Imgui Window :D");

        //<<<<<<< HEAD
        //        ImGui::Checkbox("Move Sun",&moveSun);
        //=======
        if (ImGui::Checkbox("debugLightShader", &debugLightShader))
        {
            DeferredLightingShaderNames n;
            if (debugLightShader)
            {
                n.directionalLightShader = "lighting/light_cascaded.glsl";
            }
            //            parentWindow.getRenderer()->lighting.loadShaders(n);
        }
        ImGui::Checkbox("fitShadowToCamera", &fitShadowToCamera);
        ImGui::Checkbox("fitNearPlaneToScene", &fitNearPlaneToScene);



        static float cascadeInterpolateRange = sun->getCascadeInterpolateRange();
        if (ImGui::InputFloat("Cascade Interpolate Range", &cascadeInterpolateRange))
        {
            sun->setCascadeInterpolateRange(cascadeInterpolateRange);
        }

        static float farPlane  = 150;
        static float nearPlane = 1;
        if (ImGui::InputFloat("Camera Far Plane", &farPlane))
            camera.setProj(60.0f, window->getAspectRatio(), nearPlane, farPlane);

        if (ImGui::InputFloat("Camera Near Plane", &nearPlane))
            camera.setProj(60.0f, window->getAspectRatio(), nearPlane, farPlane);

        static int numCascades = 1;

        static int currentItem      = 2;
        static const char* items[5] = {"128", "256", "512", "1024", "2048"};

        static int itemsi[5] = {128, 256, 512, 1024, 2048};

        if (ImGui::InputInt("Num Cascades", &numCascades))
        {
            sun->createShadowMap(itemsi[currentItem], itemsi[currentItem], numCascades);
        }

        if (ImGui::Combo("Shadow Map Resolution", &currentItem, items, 5))
        {
            //            state = static_cast<State>(currentItem);
            //            switchToState(static_cast<State>(currentItem));
            sun->createShadowMap(itemsi[currentItem], itemsi[currentItem], numCascades);
        }
        ImGui::End();
    }
}


int main(int argc, char* args[])
{
    // This should be only called if this is a sample located in saiga/samples
    initSaigaSample();

    Sample window;
    window.run();

    return 0;
}
