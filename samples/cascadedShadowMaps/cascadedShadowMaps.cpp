/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "cascadedShadowMaps.h"

#include "saiga/rendering/deferred_renderer.h"
#include "saiga/opengl/shader/shaderLoader.h"

#include "saiga/geometry/triangle_mesh_generator.h"

SimpleWindow::SimpleWindow(OpenGLWindow *window): Program(window)
{
    //this simplifies shader debugging
    ShaderLoader::instance()->addLineDirectives = true;

    //create a perspective camera
    float aspect = window->getAspectRatio();
    camera.setProj(60.0f,aspect,1.0f,150.0f);
    //    camera.setView(vec3(0,5,10),vec3(0,5,0),vec3(0,1,0));
    camera.setView(vec3(0,10,-10),vec3(0,9,0),vec3(0,1,0));
    camera.enableInput();
    //How fast the camera moves
    camera.movementSpeed = 2;
    camera.movementSpeedFast = 20;

    //Set the camera from which view the scene is rendered
    window->setCamera(&camera);


    //add this object to the keylistener, so keyPressed and keyReleased will be called
    SDL_EventHandler::addKeyListener(this);

    //This simple AssetLoader can create assets from meshes and generate some generic debug assets
    AssetLoader2 assetLoader;

    float height = 20;
    //First create the triangle mesh of a cube
    auto cubeMesh = TriangleMeshGenerator::createMesh(AABB(vec3(-1,0,-1),vec3(1,height,1)));

    //To render a triangle mesh we need to wrap it into an asset. This creates the required OpenGL buffers and provides
    //render functions.
    auto cubeAsset = assetLoader.assetFromMesh(cubeMesh,Colors::blue);

    float s = 200;

    for(int i = 0 ;i < 500; ++i){
        SimpleAssetObject cube;
        cube.asset = cubeAsset;
        cube.translateGlobal(glm::linearRand(vec3(-s,0,-s),vec3(s,0,s)));
        cube.calculateModel();
        cubes.push_back(cube);
    }

    sceneBB = AABB(vec3(-s,0,-s),vec3(s,height,s));

    groundPlane.asset = assetLoader.loadDebugPlaneAsset(vec2(s,s),1.0f,Colors::lightgray,Colors::gray);

    //create one directional light
    sun = window->getRenderer()->lighting.createDirectionalLight();
    sun->setDirection(vec3(-1,-2,-2.5));
    //    sun->setDirection(vec3(0,-1,0));
    sun->setColorDiffuse(LightColorPresets::DirectSunlight);
    sun->setIntensity(1.0);
    sun->setAmbientIntensity(0.02f);
    sun->createShadowMap(512,512);
    sun->enableShadows();


    cout<<"Program Initialized!"<<endl;
}

SimpleWindow::~SimpleWindow()
{
    //We don't need to delete anything here, because objects obtained from saiga are wrapped in smart pointers.
}

void SimpleWindow::update(float dt){
    //Update the camera position
    camera.update(dt);
    if(fitShadowToCamera){
        sun->fitShadowToCamera(&camera);
    }
    if(fitNearPlaneToScene){
        sun->fitNearPlaneToScene(sceneBB);
    }
}

void SimpleWindow::interpolate(float dt, float interpolation) {
    //Update the camera rotation. This could also be done in 'update' but
    //doing it in the interpolate step will reduce latency
    camera.interpolate(dt,interpolation);
}

void SimpleWindow::render(Camera *cam)
{
    //Render all objects from the viewpoint of 'cam'
    groundPlane.render(cam);

    for(auto &c : cubes){
        c.render(cam);
    }
}

void SimpleWindow::renderDepth(Camera *cam)
{
    //Render the depth of all objects from the viewpoint of 'cam'
    //This will be called automatically for shadow casting light sources to create shadow maps
    groundPlane.renderDepth(cam);

    for(auto &c : cubes){
        c.renderDepth(cam);
    }
}

void SimpleWindow::renderOverlay(Camera *cam)
{
    //The skybox is rendered after lighting and before post processing
    skybox.render(cam);
}

void SimpleWindow::renderFinal(Camera *cam)
{


    {
        ImGui::SetNextWindowPos(ImVec2(50, 50), ImGuiSetCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(400,200), ImGuiSetCond_FirstUseEver);
        ImGui::Begin("An Imgui Window :D");

//<<<<<<< HEAD
//        ImGui::Checkbox("Move Sun",&moveSun);
//=======
        if(ImGui::Checkbox("debugLightShader",&debugLightShader)){
            DeferredLightingShaderNames n;
            if(debugLightShader){
                n.directionalLightShader = "lighting/light_cascaded.glsl";
            }
            parentWindow->getRenderer()->lighting.loadShaders(n);
        }
        ImGui::Checkbox("fitShadowToCamera",&fitShadowToCamera);
        ImGui::Checkbox("fitNearPlaneToScene",&fitNearPlaneToScene);



        static float cascadeInterpolateRange = sun->getCascadeInterpolateRange();
        if(ImGui::InputFloat("Cascade Interpolate Range",&cascadeInterpolateRange)){
            sun->setCascadeInterpolateRange(cascadeInterpolateRange);
        }

        static float farPlane = 150;
        static float nearPlane = 1;
        if(ImGui::InputFloat("Camera Far Plane",&farPlane))
            camera.setProj(60.0f,parentWindow->getAspectRatio(),nearPlane,farPlane);

        if(ImGui::InputFloat("Camera Near Plane",&nearPlane))
            camera.setProj(60.0f,parentWindow->getAspectRatio(),nearPlane,farPlane);

        static int numCascades = 1;

        static int currentItem = 2;
        static const char *items[5] = {
            "128",
            "256",
            "512",
            "1024",
            "2048"
        };

        static int itemsi[5] = {
            128,
            256,
            512,
            1024,
            2048
        };

        if(ImGui::InputInt("Num Cascades",&numCascades)){
            sun->createShadowMap(itemsi[currentItem],itemsi[currentItem],numCascades);
        }

        if(ImGui::Combo("Shadow Map Resolution",&currentItem,items,5)){
//            state = static_cast<State>(currentItem);
//            switchToState(static_cast<State>(currentItem));
            sun->createShadowMap(itemsi[currentItem],itemsi[currentItem],numCascades);
        }
        ImGui::End();
    }

    parentWindow->renderImGui();

}

#include <fstream>

void SimpleWindow::keyPressed(SDL_Keysym key)
{
    switch(key.scancode){
    case SDL_SCANCODE_T:
    {
        std::vector<uint8_t> binary;
        GLenum format;
         parentWindow->getRenderer()->blitDepthShader->getBinary(binary,format);
         parentWindow->getRenderer()->blitDepthShader->setBinary(binary,format);
         cout << "binary size: " << binary.size() << endl;

         std::ofstream outfile ("binary.txt",std::ofstream::binary);
         outfile.write( (char*)binary.data(),binary.size());
        break;
    }
    case SDL_SCANCODE_ESCAPE:
        parentWindow->close();
        break;
    case SDL_SCANCODE_BACKSPACE:
        parentWindow->getRenderer()->printTimings();
        break;
    case SDL_SCANCODE_R:
        ShaderLoader::instance()->reload();
        break;
    case SDL_SCANCODE_F12:
        parentWindow->screenshot("screenshot.png");
        break;
    default:
        break;
    }
}

void SimpleWindow::keyReleased(SDL_Keysym key)
{
}



