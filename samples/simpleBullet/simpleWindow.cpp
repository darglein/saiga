/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "simpleWindow.h"

#include "saiga/rendering/deferred_renderer.h"
#include "saiga/opengl/shader/shaderLoader.h"

#include "saiga/geometry/triangle_mesh_generator.h"
#include "saiga/imgui/imgui.h"

#include "saiga/assets/plyLoader.h"

SimpleWindow::SimpleWindow(OpenGLWindow *window): Program(window)
{
    //create a perspective camera
    float aspect = window->getAspectRatio();
    camera.setProj(60.0f,aspect,0.1f,50.0f);
    camera.setView(vec3(0,5,10),vec3(0,0,0),vec3(0,1,0));
    camera.enableInput();
    //How fast the camera moves
    camera.movementSpeed = 10;
    camera.movementSpeedFast = 20;

    //Set the camera from which view the scene is rendered
    window->setCamera(&camera);


    //add this object to the keylistener, so keyPressed and keyReleased will be called
    SDL_EventHandler::addKeyListener(this);

    //This simple AssetLoader can create assets from meshes and generate some generic debug assets
    AssetLoader assetLoader;

    //First create the triangle mesh of a cube
    auto cubeMesh = TriangleMeshGenerator::createMesh(AABB(vec3(-1),vec3(1)));




    cubeAsset = assetLoader.assetFromMesh(*cubeMesh,Colors::blue);




    groundPlane.asset = assetLoader.loadDebugPlaneAsset(vec2(20,20),1.0f,Colors::lightgray,Colors::gray);

    //create one directional light
    sun = window->getRenderer()->lighting.createDirectionalLight();
    sun->setDirection(vec3(-1,-3,-2));
    sun->setColorDiffuse(LightColorPresets::DirectSunlight);
    sun->setIntensity(1.0);
    sun->setAmbientIntensity(0.3f);
    sun->createShadowMap(2048,2048);
    sun->enableShadows();

    initBullet();
    cout<<"Program Initialized!"<<endl;
}

SimpleWindow::~SimpleWindow()
{
    //We don't need to delete anything here, because objects obtained from saiga are wrapped in smart pointers.
}

void SimpleWindow::initBullet()
{

    {
        btCollisionShape* groundShape = new btBoxShape(btVector3(btScalar(20.), btScalar(20.), btScalar(20.)));

        physics.collisionShapes.push_back(groundShape);

        btRigidBody* body = createRigidBody(groundShape,0,vec3(0,-20,0));

        //add the body to the dynamics world
        physics.dynamicsWorld.addRigidBody(body);
    }

    {
        //create a dynamic rigidbody

        btCollisionShape* colShape = new btBoxShape(btVector3(1,1,1));
        //btCollisionShape* colShape = new btSphereShape(btScalar(1.));
        physics.collisionShapes.push_back(colShape);


        for(int i =0; i < 50; ++i)
        {
            btRigidBody* body = createRigidBody(colShape,1,vec3(0, 10, 0) + glm::linearRand(vec3(-5,0,-5),vec3(5,30,5)));
            physics.dynamicsWorld.addRigidBody(body);

            PhysicAssetObject pao;
            pao.rigidBody = body;
            pao.asset = cubeAsset;
            cubes.push_back(pao);
        }

    }


}

void SimpleWindow::update(float dt){
    //Update the camera position
    camera.update(dt);
    sun->fitShadowToCamera(&camera);

    physics.update();

    for(auto& objects : cubes)
    {
        objects.loadFromRigidbody();
        objects.calculateModel();
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

    for(auto& cube : cubes)
    cube.render(cam);
}

void SimpleWindow::renderDepth(Camera *cam)
{
    //Render the depth of all objects from the viewpoint of 'cam'
    //This will be called automatically for shadow casting light sources to create shadow maps
    groundPlane.renderDepth(cam);
    for(auto& cube : cubes)
    cube.renderDepth(cam);
}

void SimpleWindow::renderOverlay(Camera *cam)
{
    //The skybox is rendered after lighting and before post processing
    skybox.render(cam);
    physics.render(cam);
}

void SimpleWindow::renderFinal(Camera *cam)
{
    //The final render path (after post processing).
    //Usually the GUI is rendered here.

    {
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiSetCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(400,200), ImGuiSetCond_FirstUseEver);
        ImGui::Begin("An Imgui Window :D");

        ImGui::End();
    }
}


void SimpleWindow::keyPressed(SDL_Keysym key)
{
    switch(key.scancode){
    case SDL_SCANCODE_ESCAPE:
        parentWindow->close();
        break;
    case SDL_SCANCODE_BACKSPACE:
        parentWindow->getRenderer()->printTimings();
        break;
    case SDL_SCANCODE_R:
        ShaderLoader::instance()->reload();
        break;
    case SDL_SCANCODE_F11:
        parentWindow->screenshotRenderDepth("depth.png");
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



