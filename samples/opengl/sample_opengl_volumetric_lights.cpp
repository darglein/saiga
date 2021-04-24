/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "saiga/core/model/model_from_shape.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/window/SampleWindowDeferred.h"


using namespace Saiga;

class Sample : public SampleWindowDeferred
{
    using Base = SampleWindowDeferred;

   public:
    Sample()
    {
        auto cubeAsset = std::make_shared<TexturedAsset>(UnifiedModel("box.obj"));

        cube1.asset = cubeAsset;
        cube2.asset = cubeAsset;
        cube1.translateGlobal(vec3(11, 1, -2));
        cube1.calculateModel();

        cube2.translateGlobal(vec3(-11, 1, 2));
        cube2.calculateModel();

        auto sphereAsset = std::make_shared<ColoredAsset>(UnifiedModel("teapot.obj").ComputeColor());
        sphere.asset     = sphereAsset;
        sphere.translateGlobal(vec3(0, 1, 8));
        sphere.rotateLocal(vec3(0, 1, 0), 180);
        sphere.calculateModel();



        pointLight = std::make_shared<PointLight>();
        renderer->lighting.AddLight(pointLight);
        //        pointLight->setAttenuation(AttenuationPresets::Quadratic);
        pointLight->attenuation = (vec3(0, 0, 5));
        pointLight->setIntensity(2);
        pointLight->setRadius(10);
        pointLight->setPosition(vec3(9, 3, 0));
        pointLight->setColorDiffuse(make_vec3(1));

        pointLight->castShadows =true;
        pointLight->volumetric = true;

        spotLight = std::make_shared<SpotLight>();
        renderer->lighting.AddLight(spotLight);
        spotLight->attenuation = (vec3(0, 0, 5));
        spotLight->setIntensity(2);
        spotLight->setRadius(8);
        spotLight->setPosition(vec3(-10, 5, 0));
        spotLight->setColorDiffuse(make_vec3(1));
        spotLight->castShadows = true;
        spotLight->volumetric = true;


        renderer->lighting.renderVolumetric = true;


        std::cout << "Program Initialized!" << std::endl;
    }

    void render(Camera* cam, RenderPass render_pass) override
    {
        Base::render(cam, render_pass);
        if (render_pass == RenderPass::Deferred || render_pass == RenderPass::Shadow)
        {
            cube1.render(cam);
            cube2.render(cam);
            sphere.render(cam);
        }
    }



   private:
    SimpleAssetObject cube1, cube2;
    SimpleAssetObject sphere;

    std::shared_ptr<PointLight> pointLight;
    std::shared_ptr<SpotLight> spotLight;
};



int main(int argc, char* args[])
{
    // This should be only called if this is a sample located in saiga/samples
    initSaigaSample();

    Sample window;
    window.run();

    return 0;
}
