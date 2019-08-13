/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "SynteticScene.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/vision/util/Random.h"
namespace Saiga
{
Scene SynteticScene::circleSphere(int numWorldPoints, int numCameras, int numImagePoints)
{
    Scene scene;


    for (int i = 0; i < numWorldPoints; ++i)
    {
        WorldPoint wp;
        wp.p = Random::ballRand(1);
        scene.worldPoints.push_back(wp);
    }

    Intrinsics4 intr(1000, 1000, 500, 500);
    scene.intrinsics.push_back(intr);

    for (int i = 0; i < numCameras; ++i)
    {
        double alpha = double(i) / numCameras;

        double cr = 3;
        Vec3 position(cr * sin(alpha * 2 * M_PI), 0, cr * cos(alpha * 2 * M_PI));

        Extrinsics extr;
        SE3 v;
        v.so3() = onb(-position.normalized(), Vec3(0, -1, 0));
        ;
        v.translation() = position;
        extr.se3        = v.inverse();
        scene.extrinsics.push_back(extr);


        SceneImage si;
        si.intr = 0;
        si.extr = i;

#if 1
        auto refs = Random::uniqueIndices(numImagePoints, numWorldPoints);
        std::sort(refs.begin(), refs.end());
        for (int j = 0; j < numImagePoints; ++j)
        {
            StereoImagePoint mip;
            mip.wp    = refs[j];
            auto p    = extr.se3 * scene.worldPoints[mip.wp].p;
            mip.point = intr.project(p);
            si.stereoPoints.push_back(mip);
        }
#else
        for (int j = 0; j < numWorldPoints; ++j)
        {
            MonoImagePoint mip;
            mip.wp    = j;
            auto p    = extr.se3 * scene.worldPoints[mip.wp].p;
            mip.point = intr.project(p);
            si.monoPoints.push_back(mip);
        }
#endif

        scene.images.push_back(si);
    }



    scene.fixWorldPointReferences();
    SAIGA_ASSERT(scene.valid());
    scene.rms();
    return scene;
}

Scene SynteticScene::circleSphere()
{
    return circleSphere(numWorldPoints, numCameras, numImagePoints);
}

void SynteticScene::imgui()
{
    ImGui::PushID(6832657);
    ImGui::Text("Syntetic Scene");
    ImGui::InputInt("numWorldPoints", &numWorldPoints);
    ImGui::InputInt("numCameras", &numCameras);
    ImGui::InputInt("numImagePoints", &numImagePoints);
    ImGui::PopID();
}
}  // namespace Saiga
