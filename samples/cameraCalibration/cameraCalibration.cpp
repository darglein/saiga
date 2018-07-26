/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "cameraCalibration.h"

#include "saiga/opengl/shader/shaderLoader.h"

#include "saiga/geometry/triangle_mesh_generator.h"
#include "saiga/imgui/imgui.h"

#include "saiga/opencv/stereoCalibration.h"

Sample::Sample(OpenGLWindow &window, Renderer &renderer)
    : Updating(window), ForwardRenderingInterface(renderer)
{
    //create a perspective camera
    float aspect = window.getAspectRatio();
    camera.setProj(60.0f,aspect,0.1f,50.0f);
    camera.setView(vec3(0,5,10),vec3(0,0,0),vec3(0,1,0));

    //Set the camera from which view the scene is rendered
    window.setCamera(&camera);


    //add this object to the keylistener, so keyPressed and keyReleased will be called


    //This simple AssetLoader can create assets from meshes and generate some generic debug assets
    AssetLoader assetLoader;
    groundPlane.asset = assetLoader.loadDebugGrid(20,20,1);



    for(int i = 0; i < 10000; ++i)
    {
        PointVertex v;
        v.position = glm::linearRand(vec3(-3),vec3(3));
        v.color = glm::linearRand(vec3(0),vec3(1));
        pointCloud.points.push_back(v);
    }
    pointCloud.updateBuffer();


#ifdef SAIGA_USE_OPENNI2
    RGBDCameraInput::CameraOptions co1;
    RGBDCameraInput::CameraOptions co2;
    co2.w = 320;
    co2.h = 240;
    rgbdCamera.open( co1,co2);
#endif

    cout<<"Program Initialized!"<<endl;
}

Sample::~Sample()
{
}

void Sample::update(float dt){
    //Update the camera position
    camera.update(dt);

    auto f = rgbdCamera.makeFrameData();
    rgbdCamera.readFrame(*f);
    cv::Mat m = ImageViewToMat(f->colorImg.getImageView());
    cv::imshow("rgbd",m);
    cv::waitKey(10);
}

void Sample::interpolate(float dt, float interpolation) {
    //Update the camera rotation. This could also be done in 'update' but
    //doing it in the interpolate step will reduce latency
    camera.interpolate(dt,interpolation);
}



void Sample::renderOverlay(Camera *cam)
{
    //The skybox is rendered after lighting and before post processing
    skybox.render(cam);

    //Render all objects from the viewpoint of 'cam'
    groundPlane.renderForward(cam);

    pointCloud.render(cam);
}

void Sample::captureDSLR(std::string outDir, int id)
{
    system("gphoto2 --capture-image-and-download --force-overwrite --keep-raw");
    auto file = outDir + "/" + std::to_string(id) + ".jpg";
    std::string mv = "mv capt0000.jpg " +  file;
    system(mv.c_str());
    cout << "saved to " << file << endl;
}


void Sample::captureRGBD(std::string outDir, int id)
{
    auto f = rgbdCamera.makeFrameData();
    rgbdCamera.readFrame(*f);
    auto file = outDir + "/" + std::to_string(id) + ".png";
    f->colorImg.save(file);
    cout << "saved to " << file << endl;
}

void Sample::renderFinal(Camera *cam)
{
    //The final render path (after post processing).
    //Usually the GUI is rendered here.

    parentWindow.renderImGui();

    {
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiSetCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(400,200), ImGuiSetCond_FirstUseEver);
        ImGui::Begin("capture");



        if(ImGui::Button("capture dslr"))
        {
            captureDSLR("out/dlsr",capture++);
        }

        if(ImGui::Button("capture rgbd color"))
        {
            captureRGBD("out/rgbd",capture++);
        }

        if(ImGui::Button("capture both"))
        {

            captureDSLR("out/stereo/dlsr",capture);
            captureRGBD("out/stereo/rgbd",capture++);
        }



        ImGui::End();
    }

    {
        ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiSetCond_FirstUseEver);
        ImGui::SetNextWindowSize(ImVec2(400,200), ImGuiSetCond_FirstUseEver);
        ImGui::Begin("calibration");

        StereoCalibration sc(cv::Size(13,6),0.03);



        ImGui::End();
    }

}


void Sample::keyPressed(SDL_Keysym key)
{
    switch(key.scancode){
    case SDL_SCANCODE_ESCAPE:
        parentWindow.close();
        break;
    case SDL_SCANCODE_1:
         captureDSLR("out/dlsr",capture++);
        break;
    case SDL_SCANCODE_2:
         captureRGBD("out/rgbd",capture++);
        break;
    case SDL_SCANCODE_SPACE:
        captureDSLR("out/stereo/dlsr",capture);
        captureRGBD("out/stereo/rgbd",capture++);
        break;
    default:
        break;
    }
}

void Sample::keyReleased(SDL_Keysym key)
{
}



