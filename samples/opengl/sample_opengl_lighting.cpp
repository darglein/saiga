/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/math/random.h"
#include "saiga/core/model/model_from_shape.h"
#include "saiga/core/util/arguments.h"
#include "saiga/opengl/window/SampleWindowDeferred.h"

using namespace Saiga;

class Sample : public SampleWindowDeferred
{
   public:
    using Base = SampleWindowDeferred;

    Sample()
    {
        Random::setSeed(23461);

        float aspect = window->getAspectRatio();
        camera.setProj(60.0f, aspect, 0.1f, 100.0f);

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


        //        auto stickMesh = TriangleMeshGenerator::BoxMesh(AABB(vec3(-0.2, 0, -0.2), vec3(0.2, 2, 0.2)));

        auto stick = std::make_shared<ColoredAsset>(
            BoxMesh(AABB(vec3(-0.2, 0, -0.2), vec3(0.2, 2, 0.2))).FlatShading().SetVertexColor(vec4(0.7, 0.7, 0.7, 1)));

        //        auto stick = assetLoader.assetFromMesh(*stickMesh, Colors::gray);

        for (int i = 0; i < 25; ++i)
        {
            SimpleAssetObject obj;
            obj.asset   = stick;
            float scale = linearRand(0.5, 1.5);
            obj.setScale(vec3(scale, scale, scale));
            obj.translateGlobal(linearRand(vec3(-s, 0, -s), vec3(s, 0, s)));
            obj.calculateModel();
            objects.push_back(obj);
        }


        renderer->lighting.removeLight(sun);

        for (int i = 0; i < 10; ++i)
        {
            auto light = std::make_shared<PointLight>();
            renderer->lighting.AddLight(light);

            light->setIntensity(10);


            light->setRadius(linearRand(5, 30));

            light->position = (linearRand(vec3(-s, 1, -s), vec3(s, 5, s)));

            light->setColorDiffuse(make_vec3(1));

            light->castShadows = true;

            point_lights.push_back(light);
        }



        for (int i = 0; i < 15; ++i)
        {
            auto light = std::make_shared<SpotLight>();
            renderer->lighting.AddLight(light);

            light->setIntensity(10);
            light->setRadius(linearRand(8, 15));
            light->castShadows = true;

            s               = 20;
            light->position = (linearRand(vec3(-s, 3, -s), vec3(s, 8, s)));

            light->setAngle(linearRand(30, 70));
            // light->setDirection(vec3(0, -5, 0) + linearRand(make_vec3(-2), make_vec3(2)));

            light->setColorDiffuse(make_vec3(1));

            light->direction = vec3(0, -1, 0);

            spot_lights.push_back(light);
        }

        for (int i = 0; i < 2; ++i)
        {
            auto light = std::make_shared<DirectionalLight>();
            renderer->lighting.AddLight(light);
            light->BuildCascades(3);
            light->castShadows = true;

            light->setAmbientIntensity(0);

            vec3 dir = Random::sphericalRand(1).cast<float>();
            if (dir.y() > 0) dir.y() *= -1;

            //            dir = vec3(-1, -1, -1);
            light->setIntensity(0.7);
            light->setDirection(dir);
            directional_lights.push_back(light);
        }

        for (auto l : point_lights)
        {
            l->active = (current_type == 0);
        }
        for (auto l : spot_lights)
        {
            l->active = (current_type == 1);
        }
        for (auto l : directional_lights)
        {
            l->active = (current_type == 2);
            l->fitNearPlaneToScene(bounding_box);
        }

        std::cout << "Program Initialized!" << std::endl;
    }


    void update(float dt) override { Base::update(dt); }
    void interpolate(float dt, float interpolation) override
    {
        Base::interpolate(dt, interpolation);
        for (auto& o : objects)
        {
            render_system.Add(o.asset.get(), o.model, RENDER_DEFAULT | RENDER_SHADOW);
        }
    }

    void render(RenderInfo render_info) override
    {
        Base::render(render_info);

        if (render_info.render_pass == RenderPass::GUI)
        {
            if (ImGui::Begin("Saiga Sample"))
            {
                static const char* types[3] = {"Point Light", "Spot Light", "Directional Light"};
                if (ImGui::Combo("Codec", &current_type, types, 3))
                {
                    for (auto l : point_lights)
                    {
                        l->active = (current_type == 0);
                    }
                    for (auto l : spot_lights)
                    {
                        l->active = (current_type == 1);
                    }
                    for (auto l : directional_lights)
                    {
                        l->active = (current_type == 2);
                        l->fitNearPlaneToScene(bounding_box);
                    }
                }
            }

            ImGui::End();
        }
    }

   private:
    std::vector<SimpleAssetObject> objects;


    int current_type = 0;
    std::vector<std::shared_ptr<PointLight>> point_lights;
    std::vector<std::shared_ptr<SpotLight>> spot_lights;
    std::vector<std::shared_ptr<DirectionalLight>> directional_lights;

    AABB bounding_box;
};

struct MyArguments : public Arguments
{
    float f = 1;
    int i   = 2;

    MyArguments() { RegisterArgument("f", f); }
};


int main(int argc, char* args[])
{
    MyArguments ma;
    ma.Load("test.ini", true);
    return 0;

    // This should be only called if this is a sample located in saiga/samples
    initSaigaSample();

    Sample window;
    window.run();

    return 0;
}
