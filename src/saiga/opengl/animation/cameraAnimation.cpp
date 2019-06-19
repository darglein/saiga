/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/config.h"
#ifdef SAIGA_USE_OPENGL
#    include "saiga/core/imgui/imgui.h"
#    include "saiga/core/math/String.h"
#    include "saiga/opengl/animation/cameraAnimation.h"

#    include <iostream>
namespace Saiga
{
#    if 0
Interpolation::Keyframe Interpolation::get(double time)
{

    int frame = Saiga::iCeil(time);

    int prevFrame = max(0,frame - 1);



    float alpha = fract(time);

    //    std::cout << "Interpolation " << prevFrame << "," << frame << " " << time << " " << alpha << std::endl;
    if(alpha == 0)
        return keyframes[frame];


    if(cubicInterpolation)
    {

        int if0 = max(0,prevFrame-1);
        int if1 = prevFrame;
        int if2 = frame;
        int if3 = min((int)keyframes.size()-1,frame+1);


        Keyframe& f0 = keyframes[if0];
        Keyframe& f1 = keyframes[if1];
        Keyframe& f2 = keyframes[if2];
        Keyframe& f3 = keyframes[if3];


        return interpolate(f0,f1,f2,f3,alpha);
    }

    else{
        Keyframe& f1 = keyframes[prevFrame];
        Keyframe& f2 = keyframes[frame];
        return interpolate(f1,f2,alpha);

    }
}

Saiga::Interpolation::Keyframe Saiga::Interpolation::interpolate(const Saiga::Interpolation::Keyframe &f1, const Saiga::Interpolation::Keyframe &f2, const Saiga::Interpolation::Keyframe &f3, const Saiga::Interpolation::Keyframe &f4, float alpha)
{

    float tau = 1;

    Keyframe res;


    float u = alpha;
    float u2 = u * u;
    float u3 = u2 * u;

    mat4 A = {
        0,2,0,0,
        -1,0,1,0,
        2,-5,4,-1,
        -1,3,-3,1
    };
    A = mat4(transpose(A));


    //        std::cout << A << std::endl;
    vec3 ps[4] = {f1.position,f2.position,f3.position,f4.position};

    vec3 ps2[4];


    for(int i = 0; i < 4; ++i)
    {
        vec3 p(0);
        for(int j = 0; j < 4; ++j)
        {
            p += A[j][i] * ps[j];
        }
        ps2[i] = p;
        //            std::cout << "p " << p << std::endl;
    }


    res.position = 0.5f * (1.f * ps2[0] + u * ps2[1] + u2 * ps2[2] + u3 * ps2[3]);
    //        res.position =  mix(f1.position,f2.position,alpha);


    res.rot =  slerp(f2.rot,f3.rot,alpha);
    return res;
}

Saiga::Interpolation::Keyframe Saiga::Interpolation::interpolate(const Saiga::Interpolation::Keyframe &f1, const Saiga::Interpolation::Keyframe &f2, float alpha)
{
    Keyframe res;
    res.position =  mix(f1.position,f2.position,alpha);
    res.rot =  slerp(f1.rot,f2.rot,alpha);
    return res;
}
#    endif


Interpolation::Keyframe Interpolation::getNormalized(double time)
{
    time = clamp(time, 0.0, 1.0);
    //    return get( (keyframes.size()-1)*time);

    vec3 p = positionSpline.getPointOnCurve(time);
    quat q = orientationSpline.getPointOnCurve(time);
    return {q, p};
}

void Interpolation::createAsset()
{
    //    if(keyframes.size() == 0)
    if (positionSpline.controlPoints.size() <= 1) return;

    std::vector<vec3> vertices;
    std::vector<GLuint> indices;



#    if 1
    // create control polygon
    for (int i = 0; i < (int)positionSpline.controlPoints.size() - 1; ++i)
    {
        auto p1 = positionSpline.controlPoints[i];
        auto p2 = positionSpline.controlPoints[i + 1];
        vertices.push_back(p1);
        vertices.push_back(p2);
        indices.push_back(vertices.size() - 2);
        indices.push_back(vertices.size() - 1);

        //            std::cout << "cp " << p1 << p2 << std::endl;
    }
    int idx = vertices.size();

    int steps = (positionSpline.controlPoints.size() - 1) * (subSamples + 1);
    for (int i = 0; i < steps; ++i)
    {
        float alpha = float(i) / (steps - 1);
        vec3 p      = positionSpline.getPointOnCurve(alpha);
        vertices.push_back(p);
    }
    for (int i = 0; i < steps - 1; ++i)
    {
        indices.push_back(idx + i + 0);
        indices.push_back(idx + i + 1);
    }


#    else


    //    createFrustumMesh(proj,vertices,indices);



    //    for(int i = 0; i < keyframes.size()-1; ++i)
    for (int i = 0; i < curve.controlPoints.size() - 1; ++i)
    {
        for (int j = (i == 0) ? -1 : 0; j < (subSamples + 1); ++j)
        {
            float alpha = (j + 1.0) / (subSamples + 1);

            float time = i + alpha;
            //            Keyframe kf = get(time);

            Keyframe kf;
            kf.position = curve.getPointOnCurve(time);
            kf.rot      = IDENTITY_QUATERNION;
            vec3 p      = kf.position;

            //            std::cout << "time " << time << " p " << p << std::endl;

            int idx = vertices.size();
            vertices.push_back(p);
            if (j != -1)
            {
                indices.push_back(idx - 4);
                indices.push_back(idx);
            }



            vertices.push_back(p + keyframeScale * (kf.rot * vec3(1, 0, 0)));
            vertices.push_back(p + keyframeScale * (kf.rot * vec3(0, 1, 0)));
            vertices.push_back(p + keyframeScale * (kf.rot * vec3(0, 0, 1)));

            indices.push_back(idx);
            indices.push_back(idx + 1);
            indices.push_back(idx);
            indices.push_back(idx + 2);
            indices.push_back(idx);
            indices.push_back(idx + 3);
        }
    }


#    endif

    AssetLoader al;
    cameraPathAsset = al.nonTriangleMesh(vertices, indices, GL_LINES, vec4(1, 0, 0, 1));
}



void Interpolation::start(Camera& cam, float totalTimeS, float dt)
{
    totalTicks = totalTimeS / dt;
    tick       = 0;

    std::cout << "Starting Camera Interpolation. " << totalTimeS << "s  dt=" << dt << " TotalTicks: " << totalTicks
              << std::endl;

    update(cam);
}

bool Interpolation::update(Camera& camera)
{
    if (tick > totalTicks) return false;

    float cameraAlpha = float(tick) / totalTicks;
    auto kf           = getNormalized(cameraAlpha);

    camera.position = make_vec4(kf.position, 1);
    camera.rot      = kf.rot;

    camera.calculateModel();
    camera.updateFromModel();

    cameraAlpha += 0.002;


    tick++;

    return true;
}

void Interpolation::updateCurve()
{
    positionSpline.controlPoints.clear();
    orientationSpline.controlPoints.clear();

    for (auto& kf : keyframes)
    {
        positionSpline.addPoint(kf.position);
        orientationSpline.addPoint(kf.rot);
    }


    positionSpline.normalize();
    orientationSpline.normalize();



    createAsset();
}

void Interpolation::render()
{
    if (cameraPathAsset && !isRunning())
    {
        cameraPathAsset->renderForward(nullptr, identityMat4());
    }
}
void Interpolation::renderGui(Camera& camera)
{
    bool changed = false;

    ImGui::PushID(326426);


    ImGui::InputFloat("dt", &dt);
    ImGui::InputFloat("totalTime", &totalTime);
    //    if(ImGui::Checkbox("cubicInterpolation",&cubicInterpolation))
    //    {
    //        changed = true;
    //    }


    ImGui::Text("Keyframe");
    if (ImGui::Button("Add"))
    {
        addKeyframe(camera.rot, camera.getPosition());
        changed = true;
    }

    ImGui::SameLine();

    if (ImGui::Button("Remove Last"))
    {
        keyframes.pop_back();
        changed = true;
    }

    ImGui::SameLine();

    if (ImGui::Button("Clear"))
    {
        keyframes.clear();
        changed = true;
    }

    if (ImGui::Button("start camera"))
    {
        start(camera, totalTime, dt);
        changed = true;
    }



    if (ImGui::Button("print keyframes"))
    {
        for (Keyframe& kf : keyframes)
        {
            std::cout << "keyframes.push_back({ quat" << kf.rot << ", vec3" << kf.position << "});" << std::endl;
        }
        std::cout << "createAsset();" << std::endl;

        keyframes.push_back({IDENTITY_QUATERNION, make_vec3(0)});
    }

    if (ImGui::CollapsingHeader("render"))
    {
        ImGui::Checkbox("visible", &visible);
        ImGui::InputInt("subSamples", &subSamples);
        ImGui::InputFloat("keyframeScale", &keyframeScale);
        if (ImGui::Button("update mesh")) changed = true;
    }

    if (ImGui::CollapsingHeader("modify"))
    {
        ImGui::InputInt("selectedKeyframe", &selectedKeyframe);

        if (ImGui::Button("keyframe to camera"))
        {
            auto kf         = keyframes[selectedKeyframe];
            camera.position = make_vec4(kf.position, 1);
            camera.rot      = kf.rot;

            camera.calculateModel();
            camera.updateFromModel();
        }

        if (ImGui::Button("update keyframe"))
        {
            keyframes[selectedKeyframe] = {camera.rot, camera.getPosition()};
            changed                     = true;
        }

        if (ImGui::Button("delete keyframe"))
        {
            keyframes.erase(keyframes.begin() + selectedKeyframe);
            changed = true;
        }

        if (ImGui::Button("insert keyframe"))
        {
            keyframes.insert(keyframes.begin() + selectedKeyframe, {camera.rot, camera.getPosition()});
            changed = true;
        }
    }


    if (changed)
    {
        updateCurve();
    }

    ImGui::PopID();
}



}  // namespace Saiga

#endif
