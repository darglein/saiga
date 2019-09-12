/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/camera/camera.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/texture/CubeTexture.h"
#include "saiga/opengl/texture/Texture.h"
#include "saiga/opengl/world/clipmap.h"
#include "saiga/opengl/world/heightmap.h"
#include "saiga/opengl/world/terrainmesh.h"

namespace Saiga
{
class SAIGA_OPENGL_API Terrain
{
   public:
    int layers;

    std::shared_ptr<TerrainShader> shader;
    std::shared_ptr<TerrainShader> deferredshader;
    std::shared_ptr<TerrainShader> depthshader;

    std::shared_ptr<Texture> texture1, texture2;

    Heightmap heightmap;
    mat4 model;

    vec3 viewPos;

    std::vector<Clipmap> clipmaps;
    //    Clipmap clipmaps[8];
    vec2 baseScale = vec2(1, 1);



    Terrain(int layers, int w, int h, float heightScale);

    bool loadHeightmap();
    void createHeightmap();

    void createMesh();

    void setDistance(float d);

    void render(Camera* cam);
    void renderDepth(Camera* cam);

    void update(const vec3& p);

   private:
    void renderintern(Camera* cam);

    void render(const IndexedVertexBuffer<Vertex, GLuint>& mesh, vec4 color, vec4 scale, vec4 fineOrigin);
};

}  // namespace Saiga
