/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/core/imgui/imgui.h"
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
        float aspect = window->getAspectRatio();
        camera.setProj(60.0f, aspect, 0.1f, 5.0f);
        camera.setView(vec3(0, 1, 2), vec3(0, 0, 0), vec3(0, 1, 0));
        camera.movementSpeed     = 0.3;
        camera.movementSpeedFast = 3;

        sun->castShadows = false;


        normalShader  = shaderLoader.load<MVPTextureShader>("geometry/texturedAsset_normal.glsl");
        textureShader = shaderLoader.load<MVPTextureShader>("asset/texturedAsset.glsl");


        //        auto asset   = assetLoader.loadTexturedAsset("box.obj");
        //        ObjAssetLoader assetLoader;
        //        auto asset   = assetLoader.loadTexturedAsset("user/sponza/sponza.obj");
        //        object.asset = asset;


        //        Load("user/sponza/sponza.obj");

        //        Load("/home/dari/Projects/pointrendering2/BlenderScenes/bedroom/bedroom.glb");
        // Load("/home/dari/Projects/pointrendering2/BlenderScenes/bedroom/iscv2.obj");
        Load("/home/dari/Projects/saiga/data/user/Bistro_v5_1/bistro3.gltf");

        //        Load("user/lost-empire/lost_empire.obj");
        //        Load("user/living_room/living_room.obj");
        //        Load("user/fireplace_room/fireplace_room.obj");
        //        Load("box.obj");



        std::cout << "Program Initialized!" << std::endl;
    }

    void Load(const std::string& file)
    {
        UnifiedModel model(file);


        std::cout << model.BoundingBox() << std::endl;
        model.Normalize();
        std::cout << model.BoundingBox() << std::endl;

        std::cout << model << std::endl;


        auto ta      = std::make_shared<TexturedAsset>(model);
        object.asset = ta;


        //        object.asset = std::make_shared<ColoredAsset>(model);
    }

    void update(float dt) override
    {
        Base::update(dt);
        if (autoRotate)
        {
            camera.mouseRotateAroundPoint(autoRotateSpeed, 0, camera.rotationPoint, up);
        }
    }


    void render(Camera* cam, RenderPass render_pass) override
    {
        if (render_pass == RenderPass::Shadow)
        {
            object.renderDepth(cam);
        }
        else if (render_pass == RenderPass::Deferred)
        {
            object.render(cam, render_pass);
        }
        else if (render_pass == RenderPass::Forward)
        {
            //            TexturedAsset* ta = dynamic_cast<TexturedAsset*>(object.asset.get());
            //            SAIGA_ASSERT(ta);

            if (renderObject)
            {
                if (renderGeometry)
                {
                    //                    ta->deferredShader = normalShader;
                }
                else
                {
                    //                    ta->deferredShader = textureShader;
                }
                //                object.render(cam);
            }

            if (renderWireframe)
            {
                glEnable(GL_POLYGON_OFFSET_LINE);
                //        glLineWidth(1);
                glPolygonOffset(0, -500);

                object.renderWireframe(cam);
                glDisable(GL_POLYGON_OFFSET_LINE);
            }
        }
        else if (render_pass == RenderPass::GUI)
        {
            ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize(ImVec2(400, 400), ImGuiCond_FirstUseEver);
            ImGui::Begin("Model Viewer");


            ImGui::Checkbox("renderGeometry", &renderGeometry);
            ImGui::Checkbox("renderWireframe", &renderWireframe);
            ImGui::Checkbox("renderObject", &renderObject);

            ImGui::InputText("File", file.data(), file.size());


            if (ImGui::Button("Load OBJ with Texture"))
            {
                //                ObjAssetLoader assetLoader;
                //                auto asset = assetLoader.loadTexturedAsset(std::string(file.data()));
                auto asset = std::make_shared<TexturedAsset>(UnifiedModel(std::string(file.data())));
                if (asset) object.asset = asset;
            }

            if (ImGui::Button("Load OBJ with Vertex Color"))
            {
                //                ObjAssetLoader assetLoader;
                //                auto asset = assetLoader.loadColoredAsset(std::string(file.data()));
                auto asset = std::make_shared<ColoredAsset>(UnifiedModel(std::string(file.data())));
                if (asset) object.asset = asset;
            }


            ImGui::End();

            ImGui::SetNextWindowPos(ImVec2(0, 400), ImGuiCond_FirstUseEver);
            ImGui::SetNextWindowSize(ImVec2(400, 200), ImGuiCond_FirstUseEver);
            ImGui::Begin("Camera");


            ImGui::Checkbox("autoRotate", &autoRotate);
            if (ImGui::Button("Set Rotation Point to Position"))
            {
                camera.rotationPoint = make_vec3(camera.position);
            }
            camera.imgui();
            ImGui::End();
        }
    }



   private:
    SimpleAssetObject object;


    vec3 up               = vec3(0, 1, 0);
    bool autoRotate       = false;
    float autoRotateSpeed = 0.5;

    std::array<char, 512> file = {0};


    bool renderObject    = true;
    bool renderWireframe = false;
    bool renderGeometry  = false;
    std::shared_ptr<MVPTextureShader> normalShader, textureShader;
};

int main(int argc, char* args[])
{
    // This should be only called if this is a sample located in saiga/samples
    initSaigaSample();
    Sample window;
    window.run();
    return 0;
}
