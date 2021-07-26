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
        groundPlane.setScale(vec3(5, 5, 5));
        groundPlane.calculateModel();
        //        showGrid   = false;
        //        showSkybox = false;

        renderer->params.hdr   = true;
        renderer->params.bloom = true;
        Random::setSeed(23461);

        float aspect = window->getAspectRatio();
        camera.setProj(60.0f, aspect, 0.1f, 500.0f);
        camera.setView(vec3(0, 2, -20), vec3(0, 0, 0), vec3(0, 1, 0));

        auto sphere = std::make_shared<ColoredAsset>(
            BoxMesh(AABB(vec3(-1, -1, -1), vec3(1, 1, 1))).FlatShading().SetVertexColor(vec4(0.7, 0.7, 0.7, 1)));

        //        sphereMesh->setColor()

        int s            = 20;
        bounding_box.min = vec3(-s, 0, -s);
        bounding_box.max = vec3(s, 10, s);

        std::vector<vec3> positions = {vec3(10, 3, 0), vec3(0, 3, 0), vec3(-10, 3, 0)};

        for (auto p : positions)
        {
            SimpleAssetObject obj;
            obj.asset = sphere;
            obj.translateGlobal(p);
            obj.calculateModel();
            objects.push_back(obj);
        }

        sun->ambientIntensity = 0;
        sun->intensity        = 1.54;
        sun->direction        = vec3(0, -1, 0);
        sun->castShadows      = false;

        // renderer->lighting.removeLight(sun);
        std::vector<float> intensities = {5, 10, 100};

        for (int i = 0; i < positions.size(); ++i)
        {
            auto light = std::make_shared<PointLight>();
            renderer->lighting.AddLight(light);

            light->setColorDiffuse(linearRand(vec3(0, 0, 0), vec3(1, 1, 1)));
            light->setIntensity(intensities[i]);


            light->setRadius(4);
            light->attenuation = vec3(0, 0, 1);

            light->position = positions[i] + vec3(0, 0, -2);
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
