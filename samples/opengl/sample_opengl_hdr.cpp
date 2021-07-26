/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/math/random.h"
#include "saiga/core/model/model_from_shape.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/window/SampleWindowDeferred.h"

using namespace Saiga;

class Sample : public SampleWindowDeferred
{
   public:
    using Base = SampleWindowDeferred;

    Sample()
    {
        groundPlane.setScale(vec3(5,5,5));
        groundPlane.calculateModel();

        renderer->params.hdr = true;
        Random::setSeed(23461);

        float aspect = window->getAspectRatio();
        camera.setProj(60.0f, aspect, 0.1f, 500.0f);

        auto sphere = std::make_shared<ColoredAsset>(
            BoxMesh(AABB(vec3(-1, -1, -1), vec3(1, 1, 1))).FlatShading().SetVertexColor(vec4(0.7, 0.7, 0.7, 1)));

        //        sphereMesh->setColor()

        int s            = 20;
        bounding_box.min = vec3(-s, 0, -s);
        bounding_box.max = vec3(s, 10, s);
        for (int i = 0; i < 25; ++i)
        {
            SimpleAssetObject obj;
            obj.asset   = sphere;
            float scale = linearRand(0.5, 1.5);
            obj.setScale(vec3(scale, scale, scale));
            obj.translateGlobal(linearRand(vec3(-s, scale, -s), vec3(s, scale, s)));
            obj.calculateModel();
            objects.push_back(obj);
        }


        renderer->lighting.removeLight(sun);

        for (int i = 0; i < 5; ++i)
        {
            auto light = std::make_shared<PointLight>();
            renderer->lighting.AddLight(light);

            light->setColorDiffuse(linearRand(vec3(0, 0, 0), vec3(1, 1, 1)));
            light->setIntensity(linearRand(5, 100));


            light->setRadius(linearRand(10, 30));
            light->attenuation = vec3(0, 0, 1);

            light->position = linearRand(vec3(-s, 1, -s), vec3(s, 5, s));
        }

        std::cout << "Program Initialized!" << std::endl;
    }


    void update(float dt) override { Base::update(dt); }


    void render(RenderInfo render_info) override
    {
        if (render_info.render_pass == RenderPass::Deferred || render_info.render_pass == RenderPass::Shadow)
        {
            Base::render(render_info);
            for (auto& o : objects)
            {
                o.render(render_info.camera, render_info.render_pass);
            }
        }

        if (render_info.render_pass == RenderPass::GUI)
        {
            if (ImGui::Begin("Saiga Sample"))
            {
            }

            ImGui::End();
        }
    }

   private:
    std::vector<SimpleAssetObject> objects;
    AABB bounding_box;
};



int main(int argc, char* args[])
{
    // This should be only called if this is a sample located in saiga/samples
    initSaigaSample();

    Sample window;
    window.run();

    return 0;
}
