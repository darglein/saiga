/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "forwardrendering.h"

#include "saiga/core/geometry/triangle_mesh_generator.h"
#include "saiga/core/imgui/imgui.h"
#include "saiga/core/math/random.h"
#include "saiga/opengl/shader/shaderLoader.h"

ImGui::IMConsole console;

Sample::Sample()
{
    for (int i = 0; i < 10000; ++i)
    {
        PointVertex v;
        v.position = linearRand(make_vec3(-3), make_vec3(3));
        v.color    = linearRand(make_vec3(0), make_vec3(1));
        pointCloud.points.push_back(v);
    }
    pointCloud.updateBuffer();

    {
        // let's just draw the coordiante axis
        PointVertex v;

        // x
        v.color    = vec3(1, 0, 0);
        v.position = vec3(-1, 0, 0);
        lineSoup.lines.push_back(v);
        v.position = vec3(1, 0, 0);
        lineSoup.lines.push_back(v);

        // y
        v.color    = vec3(0, 1, 0);
        v.position = vec3(0, -1, 0);
        lineSoup.lines.push_back(v);
        v.position = vec3(0, 1, 0);
        lineSoup.lines.push_back(v);

        // z
        v.color    = vec3(0, 0, 1);
        v.position = vec3(0, 0, -1);
        lineSoup.lines.push_back(v);
        v.position = vec3(0, 0, 1);
        lineSoup.lines.push_back(v);

        lineSoup.translateGlobal(vec3(5, 5, 5));
        lineSoup.calculateModel();

        lineSoup.lineWidth = 3;
        lineSoup.updateBuffer();
    }


    //    frustum.vertices.resize(2);
    //    frustum.vertices[0].position = vec4(0, 0, 0, 0);
    //    frustum.vertices[1].position = vec4(10, 10, 10, 0);
    //    frustum.fromLineList();

    //    frustum.createGrid({100, 100}, {0.1, 0.1});
    frustum.createFrustum(camera.proj, 1);
    frustum.setColor(vec4{0, 1, 0, 1});

    auto shader = shaderLoader.load<MVPShader>("colored_points.glsl");
    frustum.create(shader, shader, shader, shader);
    frustum.loadDefaultShaders();
    //    frustum.model.createFrustum(camera.proj);
    //    assetLoader.frustumMesh();



    std::cout << "Program Initialized!" << std::endl;
}



void Sample::renderOverlay(Camera* cam)
{
    Base::renderOverlay(cam);

    pointCloud.render(cam);

    lineSoup.render(cam);

    frustum.renderForward(cam, mat4::Identity());
}



void Sample::renderFinal(Camera* cam)
{
    Base::renderFinal(cam);

    ImGui::SetNextWindowPos(ImVec2(0, 0), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(200, 200), ImGuiCond_FirstUseEver);
    ImGui::Begin("test");

    if (ImGui::Button("add"))
    {
        console << "asdf " << 234 << std::endl;
    }


    ImGui::End();



    console.render();
}
int main(const int argc, const char* argv[])
{
    using namespace Saiga;

    {
        Sample example;

        example.run();
    }

    return 0;
}
