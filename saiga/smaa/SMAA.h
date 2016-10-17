#pragma once

#include "saiga/opengl/vertex.h"
#include "saiga/opengl/query/gpuTimer.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/framebuffer.h"
#include "saiga/rendering/gbuffer.h"
#include "saiga/opengl/indexedVertexBuffer.h"

class SMAA{
public:

    //RGBA temporal render targets
    Texture* edgesTex;
    Texture* blendTex;

    //supporting precalculated textures
    Texture* areaTex;
    Texture* searchTex;


    IndexedVertexBuffer<VertexNT,GLushort> quadMesh;
    vec2 screenSize;

    SMAA(int w, int h);
};
