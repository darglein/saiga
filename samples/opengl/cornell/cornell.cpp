/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "cornell.h"

Sample::Sample()
{
    // This simple AssetLoader can create assets from meshes and generate some generic debug assets
    ObjAssetLoader assetLoader;
    teapot.asset = assetLoader.loadBasicAsset("models/Cornell.obj");
    //    teapot.asset = assetLoader.loadTexturedAsset("models/box.obj");
    teapot.translateGlobal(vec3(0, 0, 0));
    teapot.calculateModel();

    showGrid = false;

    sun->setActive(false);

    float aspect = window->getAspectRatio();
    camera.setProj(35.0f, aspect, 0.1f, 100.0f);
    camera.position = vec4(0, 1, 4.5, 1);
    camera.rot      = quat::Identity();
    std::cout << "Program Initialized!" << std::endl;


    pointLight = renderer->lighting.createPointLight();
    pointLight->setAttenuation(AttenuationPresets::Quadratic);
    pointLight->setIntensity(1);
    pointLight->setRadius(3);
    pointLight->setPosition(vec3(0, 1.5, 0));
    pointLight->setColorDiffuse(make_vec3(1));
    pointLight->calculateModel();
    //        pointLight->createShadowMap(256,256,sq);
    pointLight->createShadowMap(1024, 1024, ShadowQuality::HIGH);
    pointLight->enableShadows();
}

void Sample::render(Camera* cam)
{
    // The sample draws the debug plane
    SampleWindowDeferred::render(cam);
    teapot.render(cam);
}

void Sample::renderDepth(Camera* cam)
{
    // Render the depth of all objects from the viewpoint of 'cam'
    // This will be called automatically for shadow casting light sources to create shadow maps
    SampleWindowDeferred::render(cam);
    teapot.renderDepth(cam);
}


int main(int argc, char* args[])
{
    // This should be only called if this is a sample located in saiga/samples
    initSaigaSample();

    Sample window;
    window.run();

    return 0;
}
