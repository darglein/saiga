#pragma once

#include "libhello/opengl/texture/texture.h"
#include "libhello/opengl/texture/cube_texture.h"
#include "libhello/opengl/indexedVertexBuffer.h"
#include "libhello/opengl/basic_shaders.h"

#include "libhello/world/heightmap.h"
#include "libhello/world/terrainmesh.h"

class TerrainShader : public MVPTextureShader{
public:
    GLuint location_ScaleFactor, location_FineBlockOrig,location_color, location_TexSizeScale; //vec4
    GLuint location_RingSize,location_ViewerPos, location_AlphaOffset, location_OneOverWidth; //vec2
    GLuint location_ZScaleFactor, location_ZTexScaleFactor; //float

     GLuint location_imageUp,location_normalMap,location_normalMapUp;

    TerrainShader(const string &multi_file) : MVPTextureShader(multi_file){}
    virtual void checkUniforms();
    virtual void uploadVP(const vec2 &pos);
    void uploadColor(const vec4 &s);
    void uploadScale(const vec4 &s);
    void uploadFineOrigin(const vec4 &s);
    void uploadTexSizeScale(const vec4 &s);
    void uploadRingSize(const vec2 &s);
    void uploadZScale(float f);
    void uploadNormalMap(raw_Texture *texture);
    void uploadImageUp(raw_Texture *texture);

    void uploadNormalMapUp(raw_Texture *texture);


};

class Terrain{
public:
    IndexedVertexBuffer<Vertex,GLuint> mesh,center;
    IndexedVertexBuffer<Vertex,GLuint> fixupv,fixuph,trim,trimi,degenerated;

    TerrainShader* shader;
//    std::vector<Texture*> texture;

    Heightmap heightmap;
    mat4 model;

    vec3 viewPos;

    Terrain();

    void createMesh(unsigned int w, unsigned int h);

    void setPosition(const vec3& p);
    void setDistance(float d);
    void render(const vec3 &viewPos, const mat4& view, const mat4 &proj);



    void render(const IndexedVertexBuffer<Vertex,GLuint> &mesh, vec4 color, vec4 scale,vec4 fineOrigin);
    void render(const IndexedVertexBuffer<Vertex,GLuint> &mesh, vec4 color, vec4 scale);

    void renderDeg(vec2 scale,vec2 cellWidth, vec4 offset,vec2 ringSize);
    void renderTrim(vec2 scale,vec2 cellWidth, vec4 offset,vec2 ringSize,float f);
    void renderFixUps(vec2 scale,vec2 cellWidth, vec4 offset,vec2 ringSize);
    void renderBlocks(vec2 scale, vec2 cellWidth, vec4 offset, vec2 ringSize);
    void renderRing(vec2 scale, float f, vec2 off);

};
