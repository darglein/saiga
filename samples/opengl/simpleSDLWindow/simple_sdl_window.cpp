/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/window/SampleWindowDeferred.h"

using namespace Saiga;

class Sample : public SampleWindowDeferred
{
    using Base = SampleWindowDeferred;

   public:
    Sample()
    {
        // This simple AssetLoader can create assets from meshes and generate some generic debug assets
        ObjAssetLoader assetLoader;
        teapot.asset = assetLoader.loadBasicAsset("models/teapot.obj");
        teapot.translateGlobal(vec3(0, 1, 0));
        teapot.calculateModel();

        std::cout << "Program Initialized!" << std::endl;
    }

    void render(Camera* cam, RenderPass render_pass) override
    {
        Base::render(cam, render_pass);
        if (render_pass == RenderPass::Deferred || render_pass == RenderPass::Shadow)
        {
            teapot.render(cam, render_pass);
        }
    }



   private:
    SimpleAssetObject teapot;
};



int main(int argc, char* args[])
{
    // This should be only called if this is a sample located in saiga/samples
    initSaigaSample();
    Sample window;
    window.run();
    return 0;
}
