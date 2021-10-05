/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/model/all.h"
#include "saiga/opengl/window/SampleWindowDeferred.h"
#include "saiga/opengl/world/skybox.h"
using namespace Saiga;

class Sample : public SampleWindowDeferred
{
    using Base = SampleWindowDeferred;

   public:
    Sample()
    {
        teapot.asset = std::make_shared<ColoredAsset>(UnifiedModel("models/teapot.obj").ComputeColor());
        teapot.translateGlobal(vec3(0, 0, 0));
        teapot.setScale(vec3(0.5, 0.5, 0.5));
        teapot.calculateModel();


        Image img("env_waterfall.jpg");
        skybox = std::make_shared<Skybox>(std::make_shared<Texture>(img));

        showGrid   = false;
        showSkybox = false;

        std::cout << "Program Initialized!" << std::endl;
    }

    void interpolate(float dt, float interpolation) override
    {
        Base::interpolate(dt, interpolation);
        render_system.Add(teapot.asset.get(), teapot.model, RENDER_DEFAULT | RENDER_SHADOW);
    }

    void render(RenderInfo render_info) override
    {
        Base::render(render_info);
        if (render_info.render_pass == RenderPass::Forward)
        {
            skybox->render(render_info.camera);
        }
    }



   private:
    SimpleAssetObject teapot;
    std::shared_ptr<Skybox> skybox;
};



int main(int argc, char* args[])
{
    // This should be only called if this is a sample located in saiga/samples
    initSaigaSample();
    Sample window;
    window.run();
    return 0;
}
