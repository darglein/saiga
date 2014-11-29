#pragma once

#include "libhello/opengl/texture/texture.h"
#include "libhello/opengl/texture/cube_texture.h"
#include "libhello/opengl/indexedVertexBuffer.h"
#include "libhello/opengl/basic_shaders.h"

#include "libhello/world/heightmap.h"

class TerrainShader : public MVPTextureShader{
public:
    GLuint location_ScaleFactor, location_FineBlockOrig,location_color; //vec4
    GLuint location_ViewerPos, location_AlphaOffset, location_OneOverWidth; //vec2
    GLuint location_ZScaleFactor, location_ZTexScaleFactor; //float
    TerrainShader(const string &multi_file) : MVPTextureShader(multi_file){}
    virtual void checkUniforms();
    virtual void uploadVP(const vec2 &pos);
    void uploadColor(const vec4 &s);
    void uploadScale(const vec4 &s);
    void uploadFineOrigin(const vec4 &s);
};

class Terrain{
public:
    IndexedVertexBuffer<Vertex,GLuint> mesh,center;
    IndexedVertexBuffer<Vertex,GLuint> fixupv,fixuph,trim,trimi;

    TerrainShader* shader;
    Texture* texture;

    Heightmap heightmap;
    mat4 model;

    vec3 viewPos;

    Terrain();

    void createMesh(unsigned int w, unsigned int h);

    void setPosition(const vec3& p);
    void setDistance(float d);
    void render(const vec3 &viewPos, const mat4& view, const mat4 &proj);

    void renderBlock(vec4 color, vec4 scale);
    void renderFixUpV(vec4 color,vec4 scale);
    void renderFixUpH(vec4 color,vec4 scale);
    void renderTrim(vec4 color,vec4 scale);
    void renderTrimi(vec4 color,vec4 scale);

    void renderCenter(vec4 color,vec2 scale);

    void renderBlocks(vec2 scale,vec2 cellWidth, vec4 offset);
    void renderRing(vec2 scale, float f, vec2 off);

};
