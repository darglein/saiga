/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "saiga/opengl/window/SampleWindowDeferred.h"
#include "saiga/opengl/world/pointCloud.h"
#include "saiga/vision/reconstruction/MeshToTSDF.h"
using namespace Saiga;

class Sample : public SampleWindowDeferred
{
    using Base = SampleWindowDeferred;

   public:
    Sample()
    {
        auto model = UnifiedModel("models/bunny.obj");

        auto triangles = model.mesh.front().TriangleSoup();
        std::cout << "num triangles " << triangles.size() << std::endl;

        std::vector<float> weights(triangles.size(), 1.f);
        SimplePointCloud points = MeshToPointCloudPoissonDisc2(triangles, weights, 100000, 0.01);
        std::cout << "num points:  " << points.size() << std::endl;

        UnifiedMesh point_mesh;
        for (auto p : points)
        {
            point_mesh.position.push_back(p.position);
            point_mesh.normal.push_back(p.normal);
            point_mesh.color.push_back(vec4(1, 1, 1, 1));
        }
        gl_point_cloud               = std::make_shared<GLPointCloud>(point_mesh);
        gl_point_cloud->point_size   = 1;
        gl_point_cloud->point_radius = 0.01;


        box.asset = std::make_shared<ColoredAsset>(model.ComputeColor());
        showGrid  = true;

        sun->active = true;

        float aspect = window->getAspectRatio();
        camera.setProj(35.0f, aspect, 0.1f, 100.0f);
        camera.position = vec4(0, 1, 4.5, 1);
        camera.rot      = quat::Identity();
        std::cout << "Program Initialized!" << std::endl;


        //        pointLight = std::make_shared<PointLight>();
        //        renderer->lighting.AddLight(pointLight);
        //        pointLight->setIntensity(1);
        //        pointLight->setRadius(3);
        //        pointLight->position = (vec3(0, 1.5, 0));
        //        pointLight->setColorDiffuse(make_vec3(1));

        //        pointLight->createShadowMap(256,256,sq);
        //        pointLight->castShadows = true;
    }


    void render(RenderInfo render_info) override
    {
        Base::render(render_info);
        if (render_info.render_pass == RenderPass::Deferred || render_info.render_pass == RenderPass::Shadow)
        {
            //            box.render(render_info.camera);
            gl_point_cloud->render(render_info);
        }

        if (render_info.render_pass == RenderPass::GUI)
        {
            gl_point_cloud->imgui();
        }
    }

   private:
    SimpleAssetObject box;
    std::shared_ptr<PointLight> pointLight;
    std::shared_ptr<GLPointCloud> gl_point_cloud;
};



int main(int argc, char* args[])
{
    // This should be only called if this is a sample located in saiga/samples
    initSaigaSample();

    Sample window;
    window.run();

    return 0;
}
