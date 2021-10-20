/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "saiga/core/model/model_from_shape.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/math/random.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/window/SampleWindowDeferred.h"

using namespace Saiga;

class Sample : public SampleWindowDeferred
{
   public:
    using Base = SampleWindowDeferred;

    Sample()
    {
        float height = 20;


        auto cubeAsset = std::make_shared<ColoredAsset>(
            BoxMesh(AABB(vec3(-1, 0, -1), vec3(1, height, 1))).SetVertexColor(vec4(0.7, 0.7, 0.7, 1)));

        float s = 200;

        for (int i = 0; i < 500; ++i)
        {
            SimpleAssetObject cube;
            cube.asset = cubeAsset;
            cube.translateGlobal(linearRand(vec3(-s, 0, -s), vec3(s, 0, s)));
            cube.calculateModel();
            cubes.push_back(cube);
        }

        sceneBB = AABB(vec3(-s, 0, -s), vec3(s, height, s));

        groundPlane.asset = std::make_shared<ColoredAsset>(
            CheckerBoardPlane(make_ivec2(20, 20), 1.0f, Colors::firebrick, Colors::gray));

        // create one directional light

        Base::sun->active = false;

        sun = std::make_shared<DirectionalLight>();
        renderer->lighting.AddLight(sun);
        sun->setDirection(vec3(-1, -2, -2.5));
        //    sun->setDirection(vec3(0,-1,0));
        sun->setColorDiffuse(LightColorPresets::DirectSunlight);
        sun->setIntensity(1.0);
        sun->setAmbientIntensity(0.02f);
        sun->BuildCascades(3);
        sun->castShadows = true;

        camera.recalculateMatrices();
        camera.recalculatePlanes();

        sun->fitShadowToCamera(&camera);
        sun->fitNearPlaneToScene(sceneBB);

        std::cout << "Program Initialized!" << std::endl;
    }
    ~Sample()
    {
        // We don't need to delete anything here, because objects obtained from saiga are wrapped in smart pointers.
    }

    void update(float dt) override
    {
        // Update the camera position
        camera.update(dt);
        if (fitShadowToCamera)
        {
            sun->fitShadowToCamera(&camera);
        }
        if (fitNearPlaneToScene)
        {
            sun->fitNearPlaneToScene(sceneBB);
        }
    }
    void interpolate(float dt, float interpolation) override
    {
        // Update the camera rotation. This could also be done in 'update' but
        // doing it in the interpolate step will reduce latency
        camera.interpolate(dt, interpolation);
    }


    void render(RenderInfo render_info) override
    {
        Base::render(render_info);
        if (render_info.render_pass == RenderPass::Deferred || render_info.render_pass == RenderPass::Shadow)
        {
            groundPlane.render(render_info.camera);

            for (auto& c : cubes)
            {
                c.render(render_info.camera);
            }
        }
        else if (render_info.render_pass == RenderPass::GUI)
        {
            ImGui::SetNextWindowPos(ImVec2(50, 50), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize(ImVec2(400, 200), ImGuiCond_FirstUseEver);
            ImGui::Begin("An Imgui Window :D");

            //<<<<<<< HEAD
            //        ImGui::Checkbox("Move Sun",&moveSun);
            //=======
            if (ImGui::Checkbox("debugLightShader", &debugLightShader))
            {
                //                DeferredLightingShaderNames n;
                //                if (debugLightShader)
                //                {
                //                    n.directionalLightShader = "lighting/light_cascaded.glsl";
                //                }
                //            parentWindow.getRenderer()->lighting.loadShaders(n);
            }
            ImGui::Checkbox("fitShadowToCamera", &fitShadowToCamera);
            ImGui::Checkbox("fitNearPlaneToScene", &fitNearPlaneToScene);



            static float cascadeInterpolateRange = sun->getCascadeInterpolateRange();
            if (ImGui::InputFloat("Cascade Interpolate Range", &cascadeInterpolateRange))
            {
                sun->setCascadeInterpolateRange(cascadeInterpolateRange);
            }

            static float farPlane  = 150;
            static float nearPlane = 1;
            if (ImGui::InputFloat("Camera Far Plane", &farPlane))
                camera.setProj(60.0f, window->getAspectRatio(), nearPlane, farPlane);

            if (ImGui::InputFloat("Camera Near Plane", &nearPlane))
                camera.setProj(60.0f, window->getAspectRatio(), nearPlane, farPlane);


//            static int currentItem      = 2;
//            static const char* items[5] = {"128", "256", "512", "1024", "2048"};
//            static int itemsi[5] = {128, 256, 512, 1024, 2048};

                        static int numCascades = 1;
            if (ImGui::InputInt("Num Cascades", &numCascades))
            {
                sun->BuildCascades(numCascades);
            }

            ImGui::End();
        }
    }

   private:
    SimpleAssetObject groundPlane;
    std::vector<SimpleAssetObject> cubes;

    AABB sceneBB;
    ProceduralSkybox skybox;


    bool debugLightShader    = false;
    bool fitShadowToCamera   = true;
    bool fitNearPlaneToScene = true;

    std::shared_ptr<DirectionalLight> sun;
};



int main(int argc, char* args[])
{
    // This should be only called if this is a sample located in saiga/samples
    initSaigaSample();

    Sample window;
    window.run();

    return 0;
}
