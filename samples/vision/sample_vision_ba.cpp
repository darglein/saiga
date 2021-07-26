/*
 * Vulkan Example - imGui (https://github.com/ocornut/imgui)
 *
 * Copyright (C) 2017 by Sascha Willems - www.saschawillems.de
 *
 * This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
 */


#include "saiga/core/image/imageTransformations.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/math/random.h"
#include "saiga/core/util/color.h"
#include "saiga/core/util/cv.h"
#include "saiga/core/util/directory.h"
#include "saiga/core/util/fileChecker.h"
#include "saiga/vision/scene/BALDataset.h"
//#include "saiga/vision/Eigen_GLM.h"
#include "saiga/opengl/window/SampleWindowForward.h"
#include "saiga/vision/ceres/CeresBA.h"
#include "saiga/vision/g2o/g2oBA2.h"
#include "saiga/vision/recursive/BAPointOnly.h"
#include "saiga/vision/recursive/BARecursive.h"
#include "saiga/vision/scene/PoseGraph.h"
#include "saiga/vision/scene/Scene.h"
#include "saiga/vision/scene/SynteticScene.h"

using namespace Saiga;

class Sample : public SampleWindowForward
{
    using Base = SampleWindowForward;

   public:
    Sample();

    void update(float dt) override;
    void render(RenderInfo render_info) override;

   private:
    // render objects
    GLPointCloud pointCloud;
    LineSoup lineSoup;
    LineVertexColoredAsset frustum;


    std::vector<vec3> boxOffsets;
    bool change = false;
    // bool uploadChanges    = true;
    float rms             = 0;
    int minMatchEdge      = 1000;
    float maxEdgeDistance = 1;
    Saiga::Object3D teapotTrans;

    Saiga::Scene scene;
    Saiga::SynteticScene::SceneCreator sscene;

    // bool displayModels = true;
    // bool showImgui     = true;

    Saiga::BAOptions baoptions;
    Saiga::BARec barec;

    std::vector<std::string> datasets;
};

Sample::Sample()
{
    sscene.numCameras     = 1;
    sscene.numWorldPoints = 3;
    sscene.numImagePoints = 2;
    scene                 = sscene.circleSphere();
    scene.addWorldPointNoise(0.01);

    Saiga::SearchPathes::data.getFiles(datasets, "vision/bal", ".txt");
    std::sort(datasets.begin(), datasets.end());
    std::cout << "Found " << datasets.size() << " BAL datasets" << std::endl;

    frustum.createFrustum(camera.proj, 0.1);
    frustum.setColor(vec4{0, 1, 0, 1});

    frustum.create();
}



void Sample::update(float dt)
{
    Base::update(dt);
    if (change)
    {
        pointCloud.points.clear();
        for (auto& wp : scene.worldPoints)
        {
            PointVertex v;
            v.position = wp.p.cast<float>();
            v.color    = linearRand(make_vec3(1), make_vec3(1));
            pointCloud.points.push_back(v);
        }
        pointCloud.updateBuffer();

        rms    = scene.rms();
        change = false;

        PoseGraph pg(scene, minMatchEdge);
        lineSoup.lines.clear();
        for (auto e : pg.edges)
        {
            if (!scene.images[e.from].valid() || !scene.images[e.to].valid()) continue;
            auto p1 = pg.vertices[e.from].Pose().inverse().translation();
            auto p2 = pg.vertices[e.to].Pose().inverse().translation();

            if ((p1 - p2).norm() > maxEdgeDistance) continue;
            //            if (p1.norm() > 10 || p2.norm() > 10) continue;

            PointVertex pc1;
            PointVertex pc2;

            pc1.position = p1.cast<float>();
            pc2.position = p2.cast<float>();

            pc1.color = vec3(0, 1, 0);
            pc2.color = pc1.color;

            lineSoup.lines.push_back(pc1);
            lineSoup.lines.push_back(pc2);
        }
        lineSoup.updateBuffer();
    }
}
void Sample::render(Camera* camera, RenderPass render_pass)
{
    if (render_pass == RenderPass::Forward)
    {
        pointCloud.render(camera);
        lineSoup.render(camera);
        for (auto& i : scene.images)
        {
            Saiga::SE3 se3 = i.se3;
            mat4 v         = (se3.matrix()).cast<float>();
            v              = Saiga::cvViewToGLView(v);
            v              = mat4(inverse(v));
            frustum.render(camera, v);
        }
    }
    else if (render_pass == RenderPass::GUI)
    {
        ImGui::SetNextWindowSize(ImVec2(200, 200), ImGuiCond_FirstUseEver);
        ImGui::Begin("Scene Loading");

        ImGui::InputInt("minMatchEdge", &minMatchEdge);
        ImGui::InputFloat("maxEdgeDistance", &maxEdgeDistance);

        sscene.imgui();
        if (ImGui::Button("Syntetic - circleSphere"))
        {
            scene  = sscene.circleSphere();
            change = true;
        }

        ImGui::Separator();


        std::vector<const char*> strings;
        for (auto& d : datasets) strings.push_back(d.data());
        static int currentItem = 0;
        ImGui::Combo("dataset", &currentItem, strings.data(), strings.size());
        if (ImGui::Button("load BAL"))
        {
            Saiga::BALDataset bald(datasets[currentItem]);
            scene = bald.makeScene();
            scene.compress();
            scene.sortByWorldPointId();
            change = true;
        }

        static std::array<char, 512> file = {0};
        ImGui::InputText("File", file.data(), file.size());

        if (ImGui::Button("load saiga scene"))
        {
            scene.load(std::string(file.data()));
            std::cout << scene << std::endl;
            change = true;
        }


        ImGui::End();



        ImGui::SetNextWindowSize(ImVec2(200, 200), ImGuiCond_FirstUseEver);
        ImGui::Begin("Vision BA Sample");



        static int its = 1;
        ImGui::SliderInt("its", &its, 0, 10);

        if (ImGui::Button("Bundle Adjust G2O"))
        {
            SAIGA_BLOCK_TIMER();
            Saiga::g2oBA2 ba;
            ba.baOptions = baoptions;
            ba.create(scene);
            ba.initAndSolve();
            change = true;
        }

        if (ImGui::Button("Bundle Adjust ceres"))
        {
            SAIGA_BLOCK_TIMER();
            Saiga::CeresBA ba;
            ba.baOptions = baoptions;
            ba.create(scene);
            ba.initAndSolve();
            change = true;
        }


        //    if (ImGui::Button("poseOnlySparse"))
        //    {
        //        SAIGA_BLOCK_TIMER();
        //        Saiga::BAPoseOnly ba;
        //        ba.poseOnlySparse(scene, its);
        //        change = true;
        //    }



        baoptions.imgui();
        if (ImGui::Button("sba recursive"))
        {
            SAIGA_BLOCK_TIMER();
            //        Saiga::BARec barec;
            barec.baOptions = baoptions;
            barec.create(scene);
            barec.initAndSolve();
            change = true;
        }

        if (ImGui::Button("posePointSparse"))
        {
            SAIGA_BLOCK_TIMER();
            Saiga::BAPointOnly ba;
            ba.baOptions = baoptions;
            ba.create(scene);
            ba.initAndSolve();
            change = true;
        }


        ImGui::End();

        ImGui::SetNextWindowSize(ImVec2(200, 200), ImGuiCond_FirstUseEver);
        ImGui::Begin("Scene");
        ImGui::Text("RMS: %f", rms);
        change |= scene.imgui();

        ImGui::End();
    }
}

int main(const int argc, const char* argv[])
{
    initSaigaSample();
    Sample example;
    example.run();
    return 0;
}
