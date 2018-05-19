/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "videoRecording.h"

#include "saiga/rendering/deferred_renderer.h"
#include "saiga/rendering/lighting/directional_light.h"
#include "saiga/opengl/shader/shaderLoader.h"

#include "saiga/geometry/triangle_mesh_generator.h"

VideoRecording::VideoRecording(OpenGLWindow *window): Program(window)
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

    ObjAssetLoader assetLoader;


    auto cubeAsset = assetLoader.loadTexturedAsset("objs/box.obj");

    cube1.asset = cubeAsset;
    cube2.asset = cubeAsset;
    cube1.translateGlobal(vec3(3,1,0));
    cube1.calculateModel();

    cube2.translateGlobal(vec3(3,1,5));
    cube2.calculateModel();

    auto sphereAsset = assetLoader.loadBasicAsset("objs/teapot.obj");
    sphere.asset = sphereAsset;
    sphere.translateGlobal(vec3(-2,1,0));
    sphere.rotateLocal(vec3(0,1,0),180);
    sphere.calculateModel();

    groundPlane.asset = assetLoader.loadDebugPlaneAsset(vec2(20,20),1.0f,Colors::lightgray,Colors::gray);

    //create one directional light
    sun = window->getRenderer()->lighting.createDirectionalLight();
    sun->setDirection(vec3(-1,-3,-2));
    sun->setColorDiffuse(LightColorPresets::DirectSunlight);
    sun->setIntensity(0.5);
    sun->setAmbientIntensity(0.1f);
    sun->createShadowMap(2048,2048);
    sun->enableShadows();


    textAtlas.loadFont("fonts/SourceSansPro-Regular.ttf",40,2,4,true);


    cout<<"Program Initialized!"<<endl;
}

VideoRecording::~VideoRecording()
{
    //We don't need to delete anything here, because objects obtained from saiga are wrapped in smart pointers.
}

void VideoRecording::update(float dt){
    //Update the camera position
    camera.update(dt);
    sun->fitShadowToCamera(&camera);

    if(encoder && frame++%frameSkip==0){
        //the encoder manages a buffer of a few frames
        auto img = encoder->getFrameBuffer();
        //read the current framebuffer to the buffer
        parentWindow->readToExistingImage(*img);
        //add an image to the video stream
        encoder->addFrame(img);
    }

    remainingFrames--;

    if(rotateCamera){
        float speed = 360.0f / 10.0 * dt;
//        float speed = 2 * glm::pi<float>();
        camera.mouseRotateAroundPoint(speed,0,vec3(0,5,0),vec3(0,1,0));
    }
    //    camera.rotateAroundPoint(vec3(0),vec3(0,1,0),1.0f);
}

void VideoRecording::interpolate(float dt, float interpolation) {
    //Update the camera rotation. This could also be done in 'update' but
    //doing it in the interpolate step will reduce latency
    camera.interpolate(dt,interpolation);
}

void VideoRecording::render(Camera *cam)
{
    //Render all objects from the viewpoint of 'cam'
    groundPlane.render(cam);
    cube1.render(cam);
    cube2.render(cam);
    sphere.render(cam);
}

void VideoRecording::renderDepth(Camera *cam)
{
    //Render the depth of all objects from the viewpoint of 'cam'
    //This will be called automatically for shadow casting light sources to create shadow maps
    groundPlane.renderDepth(cam);
    cube1.renderDepth(cam);
    cube2.renderDepth(cam);
    sphere.render(cam);
}

void VideoRecording::renderOverlay(Camera *cam)
{
    //The skybox is rendered after lighting and before post processing
    skybox.render(cam);
}

void VideoRecording::renderFinal(Camera *cam)
{

    //The final render path (after post processing).
    //Usually the GUI is rendered here.



    {
        ImGui::SetNextWindowPos(ImVec2(50, 400), ImGuiSetCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(400,200), ImGuiSetCond_FirstUseEver);
        ImGui::Begin("An Imgui Window :D");

        int w = parentWindow->getRenderer()->width;
        int h = parentWindow->getRenderer()->height;

        static int outW = w;
        static int outH = h;
        static int bitRate = 4000000;
        static char file[256] = "test.mp4";
        static int frameRateId = 0;
        static int codecId = 0;
        static int maxTimeSeconds = 10;

        ImGui::InputInt("Output Width",&outW);

        ImGui::InputInt("Output Height",&outH);
        ImGui::InputInt("Bitrate",&bitRate);


        static const char *fpsitems[3] = {
            "15",
            "30",
            "60"
        };
        ImGui::Combo("Framerate",&frameRateId,fpsitems,3);

        int frameRate = 30;
        switch(frameRateId){
        case 0:
            frameRate = 15;
            frameSkip = 4;
            break;
        case 1:
            frameRate = 30;
            frameSkip = 2;
            break;
        case 2:
            frameRate = 60;
            frameSkip = 1;
            break;
        }


        static const char *codecitems[4] = {
            "AV_CODEC_ID_H264",
            "AV_CODEC_ID_MPEG2VIDEO",
            "AV_CODEC_ID_MPEG4",
            "AV_CODEC_ID_RAWVIDEO"
        };
        ImGui::Combo("Codec",&codecId,codecitems,4);

        AVCodecID codec = AV_CODEC_ID_H264;
        switch(codecId){
        case 0:
            codec = AV_CODEC_ID_H264;
            break;
        case 1:
            codec = AV_CODEC_ID_MPEG2VIDEO;
            break;
        case 2:
            codec = AV_CODEC_ID_MPEG4;
            break;
        case 3:
            codec = AV_CODEC_ID_RAWVIDEO;
            break;

        }

        ImGui::InputInt("Max Video Length in seconds",&maxTimeSeconds);

        ImGui::InputText("Output File",file,256);

        if(!encoder){
            if(ImGui::Button("Start Recording")){
                SAIGA_ASSERT(!encoder);
                remainingFrames = maxTimeSeconds * 60;
                encoder = std::make_shared<FFMPEGEncoder>();

                encoder->startEncoding(file,outW,outH,w,h,frameRate,bitRate,codec);
            }
        }


        if(encoder){
            if(remainingFrames <= 0 || ImGui::Button("Stop Recording")){
                SAIGA_ASSERT(encoder);
                encoder->finishEncoding();
                encoder.reset();
            }
        }

        ImGui::Checkbox("Rotate Camera",&rotateCamera);

        ImGui::End();
    }



}


void VideoRecording::keyPressed(SDL_Keysym key)
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
    case SDL_SCANCODE_F12:
        parentWindow->screenshot("screenshot.png");
        break;
    default:
        break;
    }
}

void VideoRecording::keyReleased(SDL_Keysym key)
{
}



