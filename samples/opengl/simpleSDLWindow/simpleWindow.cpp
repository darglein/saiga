/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "simpleWindow.h"

Sample::Sample()
{
    // This simple AssetLoader can create assets from meshes and generate some generic debug assets
    ObjAssetLoader assetLoader;
    //    teapot.asset = assetLoader.loadBasicAsset("models/teapot.obj");
    teapot.asset = assetLoader.loadTexturedAsset("models/box.obj");
    teapot.translateGlobal(vec3(0, 1, 0));
    teapot.calculateModel();

    std::cout << "Program Initialized!" << std::endl;
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
