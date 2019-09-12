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
#include "saiga/opengl/world/heightmap.h"
#include "saiga/opengl/world/terrainmesh.h"

namespace Saiga
{
class SAIGA_OPENGL_API TerrainShader : public MVPTextureShader
{
   public:
    GLint location_ScaleFactor, location_FineBlockOrig, location_color, location_TexSizeScale;  // vec4
    GLint location_RingSize, location_ViewerPos, location_AlphaOffset, location_OneOverWidth;   // vec2
    GLint location_ZScaleFactor, location_ZTexScaleFactor;                                      // float

    GLint location_imageUp, location_normalMap, location_normalMapUp;

    GLint location_texture1, location_texture2;

    virtual void checkUniforms();
    virtual void uploadVP(const vec2& pos);
    void uploadColor(const vec4& s);
    void uploadScale(const vec4& s);
    void uploadFineOrigin(const vec4& s);
    void uploadTexSizeScale(const vec4& s);
    void uploadRingSize(const vec2& s);
    void uploadZScale(float f);
    void uploadNormalMap(std::shared_ptr<TextureBase> texture);
    void uploadImageUp(std::shared_ptr<TextureBase> texture);

    void uploadNormalMapUp(std::shared_ptr<TextureBase> texture);
    void uploadTexture1(std::shared_ptr<TextureBase> texture);
    void uploadTexture2(std::shared_ptr<TextureBase> texture);
};

class Clipmap
{
   public:
    enum State
    {
        SW = 0,
        SE = 1,
        NW = 2,
        NE = 3
    };

   public:
    static IndexedVertexBuffer<Vertex, GLuint> fixupv, fixuph, degenerated;
    static IndexedVertexBuffer<Vertex, GLuint> trimSW, trimSE, trimNW, trimNE;



    int m;
    vec2 off, scale;

    vec2 cellWidth;
    vec4 offset;
    vec2 ringSize;
    //    float f;
    vec2 vp;

    State state;

    bool colored = true;

    Clipmap *next, *previous;

   public:
    static std::shared_ptr<TerrainShader> shader;
    static IndexedVertexBuffer<Vertex, GLuint> mesh, center;

    static void createMeshes();



    void init(int m, vec2 off, vec2 scale, State state, Clipmap* next, Clipmap* previous);
    void update(const vec3& p);

    void calculatePosition(vec2 pos);

    void renderRing();

   private:
    void render(const IndexedVertexBuffer<Vertex, GLuint>& mesh, vec4 color, vec4 scale, vec4 fineOrigin);
    //    void render(const IndexedVertexBuffer<Vertex,GLuint> &mesh, vec4 color, vec4 scale);

    void renderDeg();
    void renderTrim();
    void renderFixUps();
    void renderBlocks();
};

}  // namespace Saiga
