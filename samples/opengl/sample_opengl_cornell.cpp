/**
 * Copyright (c) 2021 Darius Rückert
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
        box.asset =
            std::make_shared<ColoredAsset>(UnifiedModel("models/Cornell.obj").ComputeColor());
        showGrid = false;

        sun->active = false;

        float aspect = window->getAspectRatio();
        camera.setProj(35.0f, aspect, 0.1f, 100.0f);
        camera.position = vec4(0, 1, 4.5, 1);
        camera.rot      = quat::Identity();
        std::cout << "Program Initialized!" << std::endl;


        pointLight = std::make_shared<PointLight>();
        renderer->lighting.AddLight(pointLight);
        pointLight->setIntensity(1);
        pointLight->setRadius(3);
        pointLight->position = (vec3(0, 1.5, 0));
        pointLight->setColorDiffuse(make_vec3(1));

        //        pointLight->createShadowMap(256,256,sq);
        pointLight->castShadows = true;
    }


    void render(RenderInfo render_info) override
    {
        Base::render(render_info);
        if (render_info.render_pass == RenderPass::Deferred || render_info.render_pass == RenderPass::Shadow)
        {
            box.render(render_info.camera);
        }
    }


   private:
    SimpleAssetObject box;
    std::shared_ptr<PointLight> pointLight;
};



int main(int argc, char* args[])
{
    // This should be only called if this is a sample located in saiga/samples
    initSaigaSample();

    Sample window;
    window.run();

    return 0;
}
