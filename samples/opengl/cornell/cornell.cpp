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
        teapot.asset = assetLoader.loadColoredAsset("models/Cornell.obj");
        //    teapot.asset = assetLoader.loadTexturedAsset("models/box.obj");
        teapot.translateGlobal(vec3(0, 0, 0));
        teapot.calculateModel();

        showGrid = false;

        sun->active = false;

        float aspect = window->getAspectRatio();
        camera.setProj(35.0f, aspect, 0.1f, 100.0f);
        camera.position = vec4(0, 1, 4.5, 1);
        camera.rot      = quat::Identity();
        std::cout << "Program Initialized!" << std::endl;


        pointLight = std::make_shared<PointLight>();
        renderer->lighting.AddLight(pointLight);
        pointLight->setAttenuation(AttenuationPresets::Quadratic);
        pointLight->setIntensity(1);
        pointLight->setRadius(3);
        pointLight->position = (vec3(0, 1.5, 0));
        pointLight->setColorDiffuse(make_vec3(1));

        //        pointLight->createShadowMap(256,256,sq);
        pointLight->createShadowMap(1024, 1024, ShadowQuality::HIGH);
    }


    void render(Camera* cam, RenderPass render_pass) override
    {
        Base::render(cam, render_pass);
        if (render_pass == RenderPass::Deferred || render_pass == RenderPass::Shadow)
        {
            teapot.render(cam);
        }
    }


   private:
    SimpleAssetObject teapot;
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
