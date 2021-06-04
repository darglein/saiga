/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "SynteticScene.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/vision/util/Random.h"
namespace Saiga
{
namespace SynteticScene
{
Scene CircleSphere(int numWorldPoints, int numCameras, int numImagePoints, bool random_sphere)
{
    Scene scene;


    for (int i = 0; i < numWorldPoints; ++i)
    {
        WorldPoint wp;
        wp.p = Random::ballRand(1);
        scene.worldPoints.push_back(wp);
    }

    IntrinsicsPinholed intr(1000, 1000, 500, 500, 0);
    scene.intrinsics.push_back(intr);

    for (int i = 0; i < numCameras; ++i)
    {
        double alpha = double(i) / numCameras;

        double cr = 3;

        Vec3 position;
        if (random_sphere)
            position = Random::sphericalRand(cr);
        else
            position = Vec3(cr * sin(alpha * 2 * M_PI), 0, cr * cos(alpha * 2 * M_PI));

        SceneImage si;
        SE3 v;

        Vec3 up         = random_sphere ? Random::sphericalRand(1) : Vec3(0, -1, 0);
        v.so3()         = onb(-position.normalized(), up);
        v.translation() = position;
        si.se3          = v.inverse();



        si.intr = 0;

#if 1
        auto refs = Random::uniqueIndices(numImagePoints, numWorldPoints);
        std::sort(refs.begin(), refs.end());
        for (int j = 0; j < numImagePoints; ++j)
        {
            StereoImagePoint mip;
            mip.wp    = refs[j];
            auto p    = si.se3 * scene.worldPoints[mip.wp].p;
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

Scene SceneCreator::circleSphere()
{
    return CircleSphere(numWorldPoints, numCameras, numImagePoints, random_sphere);
}

void SceneCreator::imgui()
{
    ImGui::PushID(6832657);
    ImGui::Text("Syntetic Scene");
    ImGui::InputInt("numWorldPoints", &numWorldPoints);
    ImGui::InputInt("numCameras", &numCameras);
    ImGui::InputInt("numImagePoints", &numImagePoints);
    ImGui::Checkbox("random_sphere", &random_sphere);
    ImGui::PopID();
}
}  // namespace SynteticScene
}  // namespace Saiga
