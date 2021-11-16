/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "cameraAnimation.h"

#include "saiga/config.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/math/String.h"

#include <iostream>
namespace Saiga
{
bool SplinePath::imgui()
{
    bool changed = false;
    ImGui::InputFloat("time_in_seconds", &time_in_seconds);
    ImGui::InputInt("frame_rate", &frame_rate);

    ImGui::Text("Keyframes");
    ImGui::SetNextItemWidth(300);
    if (ImGui::ListBoxHeader("###keyfrmeas", 10))
    {
        for (int i = 0; i < keyframes.size(); ++i)
        {
            std::string str =
                std::to_string(i) + ": " + std::to_string(keyframes[i].user_index) + " " + keyframes[i].name;
            if (ImGui::Selectable(str.c_str(), selectedKeyframe == i))
            {
                selectedKeyframe = i;
            }
        }
        ImGui::ListBoxFooter();
    }
    if (ImGui::Button("remove selected"))
    {
        if (selectedKeyframe >= 0 && selectedKeyframe < keyframes.size())
        {
            changed = true;
            keyframes.erase(keyframes.begin() + selectedKeyframe);
        }
    }
    ImGui::SameLine();
    if (ImGui::Button("remove all"))
    {
        keyframes.clear();
        changed = true;
    }

    if (ImGui::Button("print"))
    {
        PrintUserId();
    }
    return changed;
}
std::vector<SplineKeyframe> SplinePath::Trajectory()
{
    int total_frames = time_in_seconds * frame_rate;
    std::vector<SplineKeyframe> result;
    for (int a = 0; a < total_frames; ++a)
    {
        double alpha = a / double(total_frames - 1);
        auto T       = spline.getPointOnCurve(alpha);
        result.push_back(T);
    }

    return result;
}
UnifiedMesh SplinePath::ProxyMesh()
{
    UnifiedMesh mesh;

    if (keyframes.size() < 4)
    {
        return mesh;
    }

    int samples = keyframes.size() * 10;

    for (int a = 0; a < samples; ++a)
    {
        double alpha = a / double(samples - 1);
        auto T       = spline.getPointOnCurve(alpha);

        mesh.position.push_back(T.pose.translation().cast<float>());
        mesh.color.push_back(vec4(1, 0, 0, 1));
    }

    for (int i = 0; i < mesh.NumVertices() - 1; ++i)
    {
        mesh.lines.push_back({i, i + 1});
    }

    return mesh;
}
void SplinePath::PrintUserId()
{
    std::vector<int> user_ids;
    for (auto& f : keyframes)
    {
        user_ids.push_back(f.user_index);
    }

    std::cout << "SplinePath User ids:\n";
    std::cout << to_string_iterator(user_ids.begin(), user_ids.end()) << std::endl;
}

// void Interpolation::updateCurve()
//{
//    positionSpline.controlPoints.clear();
//    orientationSpline.controlPoints.clear();
//
//    for (auto& kf : keyframes)
//    {
//        positionSpline.addPoint(kf.position);
//        orientationSpline.addPoint(kf.rot);
//    }
//
//
//    positionSpline.normalize();
//    orientationSpline.normalize();
//
//}
//
// void Interpolation::renderGui(Camera& camera)
//{
//    bool changed = false;
//
//    ImGui::PushID(326426);
//
//
//    ImGui::InputFloat("dt", &dt);
//    ImGui::InputFloat("totalTime", &totalTime);
//    //    if(ImGui::Checkbox("cubicInterpolation",&cubicInterpolation))
//    //    {
//    //        changed = true;
//    //    }
//
//
//    ImGui::Text("Keyframe");
//    if (ImGui::Button("Add"))
//    {
//        addKeyframe(camera.rot, camera.getPosition());
//        changed = true;
//    }
//
//    ImGui::SameLine();
//
//    if (ImGui::Button("Remove Last"))
//    {
//        keyframes.pop_back();
//        changed = true;
//    }
//
//    ImGui::SameLine();
//
//    if (ImGui::Button("Clear"))
//    {
//        keyframes.clear();
//        changed = true;
//    }
//
//    if (ImGui::Button("start camera"))
//    {
//        start(camera, totalTime, dt);
//        changed = true;
//    }
//
//
//
//    if (ImGui::Button("print keyframes"))
//    {
//        for (Keyframe& kf : keyframes)
//        {
//            std::cout << "keyframes.push_back({ quat" << kf.rot << ", vec3" << kf.position << "});" << std::endl;
//        }
//        std::cout << "createAsset();" << std::endl;
//
//        keyframes.push_back({quat::Identity(), make_vec3(0)});
//    }
//
//    if (ImGui::CollapsingHeader("render"))
//    {
//        ImGui::Checkbox("visible", &visible);
//        ImGui::InputInt("subSamples", &subSamples);
//        ImGui::InputFloat("keyframeScale", &keyframeScale);
//        if (ImGui::Button("update mesh")) changed = true;
//    }
//
//    if (ImGui::CollapsingHeader("modify"))
//    {
//        ImGui::InputInt("selectedKeyframe", &selectedKeyframe);
//
//        if (ImGui::Button("keyframe to camera"))
//        {
//            auto kf         = keyframes[selectedKeyframe];
//            camera.position = make_vec4(kf.position, 1);
//            camera.rot      = kf.rot;
//
//            camera.calculateModel();
//            camera.updateFromModel();
//        }
//
//        if (ImGui::Button("update keyframe"))
//        {
//            keyframes[selectedKeyframe] = {camera.rot, camera.getPosition()};
//            changed                     = true;
//        }
//
//        if (ImGui::Button("delete keyframe"))
//        {
//            keyframes.erase(keyframes.begin() + selectedKeyframe);
//            changed = true;
//        }
//
//        if (ImGui::Button("insert keyframe"))
//        {
//            keyframes.insert(keyframes.begin() + selectedKeyframe, {camera.rot, camera.getPosition()});
//            changed = true;
//        }
//    }
//
//
//    if (changed)
//    {
//        updateCurve();
//    }
//
//    ImGui::PopID();
//}
//


}  // namespace Saiga
