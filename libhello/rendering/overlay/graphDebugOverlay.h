#pragma once

#include "libhello/util/glm.h"
#include "libhello/opengl/indexedVertexBuffer.h"
#include "libhello/rendering/object3d.h"
#include <vector>

class MVPColorShader;

class SAIGA_GLOBAL GraphDebugOverlay: public Object3D {

    struct Graph{
        std::vector<float> data;
        float lastDataPoint = 0;
        VertexBuffer<Vertex> buffer;
        vec4 color = vec4(1,1,1,1);



        Graph(){
            static int id = 0;

            switch(id){
            case 0:
                color = vec4(1,1,0,1);
                break;
            case 1:
                color = vec4(1,0,0,1);
                break;

            case 2:
                color = vec4(0,1,0,1);
                break;
            case 3:
                color = vec4(0,0,1,1);
                break;
            }

            id++;
        }

        //no copying
        //Graph (const Graph&) = delete;
       // Graph& operator=(const Graph&) = delete;

    };

private:

    mat4 proj;

    int width,height;


    std::vector<Graph> graphs;


public:

    MVPColorShader* shader;

    VertexBuffer<Vertex> borderBuffer;

    GraphDebugOverlay(int width, int height, int numGraphs, int numDataPoints);

    void setFrameData(float dataPoint, int graph);
    void setScreenPosition(vec2 start, vec2 end);

    void update();
    void render(float interpolation);

};


