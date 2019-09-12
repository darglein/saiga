/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "volumetricLights.h"

#include "saiga/core/geometry/triangle_mesh_generator.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/opengl/shader/shaderLoader.h"

Sample::Sample()
{
    ObjAssetLoader assetLoader;


    auto cubeAsset = assetLoader.loadTexturedAsset("box.obj");

    cube1.asset = cubeAsset;
    cube2.asset = cubeAsset;
    cube1.translateGlobal(vec3(11, 1, -2));
    cube1.calculateModel();

    cube2.translateGlobal(vec3(-11, 1, 2));
    cube2.calculateModel();

    auto sphereAsset = assetLoader.loadBasicAsset("teapot.obj");
    sphere.asset     = sphereAsset;
    sphere.translateGlobal(vec3(0, 1, 8));
    sphere.rotateLocal(vec3(0, 1, 0), 180);
    sphere.calculateModel();


    ShadowQuality sq = ShadowQuality::HIGH;

    pointLight = renderer->lighting.createPointLight();
    //        pointLight->setAttenuation(AttenuationPresets::Quadratic);
    pointLight->setAttenuation(vec3(0, 0, 5));
    pointLight->setIntensity(2);
    pointLight->setRadius(10);
    pointLight->setPosition(vec3(9, 3, 0));
    pointLight->setColorDiffuse(make_vec3(1));
    pointLight->calculateModel();
    //        pointLight->createShadowMap(256,256,sq);
    pointLight->createShadowMap(512, 512, sq);
    pointLight->enableShadows();
    pointLight->setVolumetric(true);

    spotLight = renderer->lighting.createSpotLight();
    spotLight->setAttenuation(vec3(0, 0, 5));
    spotLight->setIntensity(2);
    spotLight->setRadius(8);
    spotLight->setPosition(vec3(-10, 5, 0));
    spotLight->setColorDiffuse(make_vec3(1));
    spotLight->calculateModel();
    spotLight->createShadowMap(512, 512, sq);
    spotLight->enableShadows();
    spotLight->setVolumetric(true);

    boxLight = renderer->lighting.createBoxLight();
    boxLight->setIntensity(1.0);

    //        boxLight->setPosition(vec3(0,2,10));
    //        boxLight->rotateLocal(vec3(1,0,0),30);
    boxLight->setView(vec3(0, 2, 10), vec3(0, 0, 13), vec3(0, 1, 0));
    boxLight->setColorDiffuse(make_vec3(1));
    boxLight->setScale(vec3(5, 5, 8));
    boxLight->calculateModel();
    boxLight->createShadowMap(512, 512, sq);
    boxLight->enableShadows();
    boxLight->setVolumetric(true);


    renderer->lighting.renderVolumetric = true;


    std::cout << "Program Initialized!" << std::endl;
}



void Sample::render(Camera* cam)
{
    Base::render(cam);
    cube1.render(cam);
    cube2.render(cam);
    sphere.render(cam);
}

void Sample::renderDepth(Camera* cam)
{
    Base::renderDepth(cam);
    cube1.renderDepth(cam);
    cube2.renderDepth(cam);
    sphere.render(cam);
}


int main(int argc, char* args[])
{
    // This should be only called if this is a sample located in saiga/samples
    initSaigaSample();

    Sample window;
    window.run();

    return 0;
}
