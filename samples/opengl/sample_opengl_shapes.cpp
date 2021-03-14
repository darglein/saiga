/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/model/all.h"
#include "saiga/core/model/model_from_shape.h"
#include "saiga/opengl/window/SampleWindowDeferred.h"
#include "saiga/opengl/window/message_box.h"
using namespace Saiga;

class Sample : public SampleWindowDeferred
{
    using Base = SampleWindowDeferred;

   public:
    Sample()
    {
        sun->setIntensity(0.7);
        sun->setAmbientIntensity(0.1);
        //        AddObject(UnifiedModel("models/teapot.obj"));
        AddObject(UVSphereMesh(Sphere(vec3(0, 0, 0), 1), 5, 5));
        AddObject(IcoSphereMesh(Sphere(vec3(0, 0, 0), 1), 3));
        AddObject(CylinderMesh(1, 3, 10));
        AddObject(ConeMesh(Cone(vec3(0, 0, 0), vec3(0, -1, 0), 1, 1), 15));
        AddObject(PlaneMesh(Plane(vec3(0, 0, 0), vec3(1, 1, 1))));
        //        AddObject(UnifiedModel("models/teapot.obj"));
        //        AddObject(UnifiedModel("models/teapot.obj"));


        std::cout << "Program Initialized!" << std::endl;
    }

    void AddObject(UnifiedModel model)
    {
        float d = 2;


        model.Normalize(1);

        int i = objects.size() % 3;
        int j = objects.size() / 3;

        SimpleAssetObject sao;
        auto co = std::make_shared<ColoredAsset>(model);
        co->setColor(vec4(1, 1, 0, 1));
        sao.asset = co;

        sao.translateGlobal(vec3(i * d, 1, j * d));
        sao.calculateModel();

        objects.push_back(sao);
    }

    void render(Camera* cam, RenderPass render_pass) override
    {
        Base::render(cam, render_pass);

        for (auto& o : objects)
        {
            o.render(cam, render_pass);
        }



        if (render_pass == RenderPass::Forward)
        {
            glEnable(GL_POLYGON_OFFSET_LINE);
            glPolygonOffset(0.0f, -500.0f);
            for (auto& o : objects)
            {
                o.renderWireframe(cam);
            }
            glDisable(GL_POLYGON_OFFSET_LINE);
        }

        if (render_pass == RenderPass::GUI)
        {
            //            ImGui::ShowDemoWindow();
        }
    }



   private:
    std::vector<SimpleAssetObject> objects;
};



int main(int argc, char* args[])
{
    // This should be only called if this is a sample located in saiga/samples
    initSaigaSample();
    Sample window;
    window.run();
    return 0;
}
