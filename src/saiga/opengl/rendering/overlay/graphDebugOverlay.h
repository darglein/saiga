/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/geometry/object3d.h"
#include "saiga/core/math/math.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/opengl/vertex.h"

#include <memory>
#include <vector>

namespace Saiga
{
class MVPColorShader;

class SAIGA_OPENGL_API GraphDebugOverlay : public Object3D
{
    struct Graph
    {
        std::vector<float> data;
        float lastDataPoint = 0;
        VertexBuffer<Vertex> buffer;
        vec4 color = vec4(1, 1, 1, 1);



        Graph()
        {
            static int id = 0;

            switch (id)
            {
                case 0:
                    color = vec4(1, 1, 0, 1);
                    break;
                case 1:
                    color = vec4(1, 0, 0, 1);
                    break;

                case 2:
                    color = vec4(0, 1, 0, 1);
                    break;
                case 3:
                    color = vec4(0, 0, 1, 1);
                    break;
            }

            id++;
        }

        // no copying
        // Graph (const Graph&) = delete;
        // Graph& operator=(const Graph&) = delete;
    };

   private:
    mat4 proj;

    int height;


    std::vector<Graph> graphs;


   public:
    std::shared_ptr<MVPColorShader> shader;

    VertexBuffer<Vertex> borderBuffer;

    GraphDebugOverlay(int width, int height, int numGraphs, int numDataPoints);

    void setFrameData(float dataPoint, int graph);
    void setScreenPosition(vec2 start, vec2 end);

    void update();
    void render(float interpolation);
};

}  // namespace Saiga
