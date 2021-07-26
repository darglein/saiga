/**
 * Copyright (c) 2021 Darius Rückert
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


        // Triangle meshes
        AddObject(UVSphereMesh(Sphere(vec3(0, 0, 0), 1), 20, 20));
        AddObject(IcoSphereMesh(Sphere(vec3(0, 0, 0), 1), 3));
        AddObject(CylinderMesh(1, 3, 10));
        AddObject(ConeMesh(Cone(vec3(0, 0, 0), vec3(0, -1, 0), 1, 1), 15));
        AddObject(PlaneMesh(Plane(vec3(0, 0, 0), vec3(1, 1, 1))));
        AddObject(BoxMesh(AABB(vec3(-1, -1, -1), vec3(1, 1, 1))));
        AddObject(SkyboxMesh(AABB(vec3(-1, -1, -1), vec3(1, 1, 1))));
        AddObject(CheckerBoardPlane(ivec2(4, 4), 0.1, Colors::blue, Colors::teal));

        // Line meshes
        AddObject(GridBoxLineMesh(AABB(vec3(-1, -1, -1), vec3(1, 1, 1)), ivec3(10, 10, 10)), true);
        AddObject(GridBoxLineMesh(AABB(vec3(-1, -1, -1), vec3(1, 1, 1)), ivec3(1, 1, 1)), true);
        AddObject(GridPlaneLineMesh(ivec2(1, 1), vec2(2, 2)), true);
        AddObject(FrustumLineMesh(camera.proj, 1, false), true);

        mat3 K  = mat3::Identity();
        K(0, 0) = 500;
        K(1, 1) = 500;
        K(0, 2) = 250;
        K(1, 2) = 250;
        AddObject(FrustumCVLineMesh(K, 1, 500, 500), true);

        camera.recalculatePlanes();
        AddObject(FrustumLineMesh(camera), true);


        std::cout << "Program Initialized!" << std::endl;
    }

    void AddObject(UnifiedMesh model, bool lines = false)
    {
        float d = 2;


        model.Normalize(1);

        int n = objects.size() + line_objects.size();
        // Place  them in a 3x3 grid
        int i = n % 3;
        int j = n / 3;

        SimpleAssetObject sao;

        if (lines)
        {
            model.SetVertexColor(vec4(1, 1, 0, 1));
            auto co = std::make_shared<LineVertexColoredAsset>(model);
            sao.asset = co;

            sao.translateGlobal(vec3(i * d, 1, j * d));
            sao.calculateModel();
            line_objects.push_back(sao);
        }
        else
        {
            model.SetVertexColor(vec4(1, 1, 0, 1));
            auto co = std::make_shared<ColoredAsset>(model);
            sao.asset = co;

            sao.translateGlobal(vec3(i * d, 1, j * d));
            sao.calculateModel();
            objects.push_back(sao);
        }
    }

    void render(RenderInfo render_info) override
    {
        Base::render(render_info);

        for (auto& o : objects)
        {
            o.render(render_info.camera, render_info.render_pass);
        }



        if (render_info.render_pass == RenderPass::Forward)
        {
            glEnable(GL_POLYGON_OFFSET_LINE);
            glPolygonOffset(0.0f, -500.0f);
            for (auto& o : objects)
            {
                o.renderWireframe(render_info.camera);
            }
            glDisable(GL_POLYGON_OFFSET_LINE);

            for (auto& o : line_objects)
            {
                o.renderForward(render_info.camera);
            }
        }

        if (render_info.render_pass == RenderPass::GUI)
        {
            //            ImGui::ShowDemoWindow();
        }
    }



   private:
    std::vector<SimpleAssetObject> objects;
    std::vector<SimpleAssetObject> line_objects;
};



int main(int argc, char* args[])
{
    // This should be only called if this is a sample located in saiga/samples
    initSaigaSample();
    Sample window;
    window.run();
    return 0;
}
