/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/rendering/overlay/graphDebugOverlay.h"

#include "saiga/core/geometry/triangle_mesh.h"
#include "saiga/core/math/random.h"
#include "saiga/opengl/framebuffer.h"
#include "saiga/opengl/shader/basic_shaders.h"

namespace Saiga
{
GraphDebugOverlay::GraphDebugOverlay(int width, int height, int numGraphs, int numDataPoints)
    : height(height), graphs(numGraphs)
{
    proj = ortho(0.0f, (float)width, 0.0f, (float)height, 1.0f, -1.0f);

    // border
    std::vector<Vertex> vertices;

    Vertex v;
    vertices.push_back(Vertex(vec3(0, 0, 0)));
    vertices.push_back(Vertex(vec3(0, 1, 0)));
    vertices.push_back(Vertex(vec3(0, 1, 0)));
    vertices.push_back(Vertex(vec3(1, 1, 0)));
    vertices.push_back(Vertex(vec3(1, 1, 0)));
    vertices.push_back(Vertex(vec3(1, 0, 0)));
    vertices.push_back(Vertex(vec3(1, 0, 0)));
    vertices.push_back(Vertex(vec3(0, 0, 0)));
    borderBuffer.set(vertices, GL_STATIC_DRAW);
    borderBuffer.setDrawMode(GL_LINES);


    // graphs
    for (int k = 0; k < numGraphs; ++k)
    {
        std::vector<Vertex> dataPoints;
        dataPoints.resize(numDataPoints);


        for (unsigned int i = 0; i < dataPoints.size(); ++i)
        {
            dataPoints[i].position = vec4(i / (float)dataPoints.size(), linearRand(0.f, 1.f), 0, 1);
        }

        graphs[k].data.resize(numDataPoints);
        graphs[k].buffer.set(dataPoints, GL_DYNAMIC_DRAW);
        graphs[k].buffer.setDrawMode(GL_LINE_STRIP);
    }



    setScreenPosition(vec2(0, 0), vec2(width, height));
}

void GraphDebugOverlay::setFrameData(float dataPoint, int graph)
{
    graphs[graph].lastDataPoint = dataPoint;
}

void GraphDebugOverlay::update()
{
    for (Graph& g : graphs)
    {
        g.data.erase(g.data.begin());
        g.data.emplace_back(g.lastDataPoint);
        float min = 99999;
        float max = 0;
        for (int i = 0; i < (int)g.data.size(); ++i)
        {
            if (g.data[i] > max)
            {
                max = g.data[i];
            }

            if (g.data[i] < min)
            {
                min = g.data[i];
            }
        }

        //    min = 0.f;

        std::vector<Vertex> dataPoints(g.data.size());

        for (int i = 0; i < (int)dataPoints.size(); ++i)
        {
            dataPoints[i].position[0] = i / (float)dataPoints.size();
            // scale to [0,1]
            dataPoints[i].position[1] = (g.data[i] - min) / (max - min);
        }

        g.buffer.updateBuffer(&dataPoints[0], dataPoints.size(), 0);
    }
}

void GraphDebugOverlay::setScreenPosition(vec2 start, vec2 end)
{
    vec2 mid = (start + end) / 2.f;
    mid[1]   = height - mid[1];

    vec2 S = vec2(start - end).array().abs();
    model  = translate(make_vec3(mid, 0) + make_vec3(-S / 2.f, 0));
    model  = model * Saiga::scale(make_vec3(S, 0));
}

void GraphDebugOverlay::render(float interpolation)
{
    (void)interpolation;
    shader->bind();

    shader->uploadModel(model);
    //    shader->uploadProj(proj);

    for (Graph& g : graphs)
    {
        shader->uploadColor(vec4(g.color));
        g.buffer.bindAndDraw();
    }


    shader->uploadColor(vec4(1, 1, 1, 1));
    borderBuffer.bindAndDraw();

    shader->unbind();
}

}  // namespace Saiga
