#pragma once

#include "libhello/opengl/texture/texture.h"
#include "libhello/opengl/texture/cube_texture.h"
#include "libhello/opengl/indexedVertexBuffer.h"
#include "libhello/opengl/basic_shaders.h"

#include "libhello/world/heightmap.h"
#include "libhello/world/terrainmesh.h"
#include "libhello/camera/camera.h"


class SAIGA_GLOBAL TerrainShader : public MVPTextureShader{
public:
    GLuint location_ScaleFactor, location_FineBlockOrig,location_color, location_TexSizeScale; //vec4
    GLuint location_RingSize,location_ViewerPos, location_AlphaOffset, location_OneOverWidth; //vec2
    GLuint location_ZScaleFactor, location_ZTexScaleFactor; //float

     GLuint location_imageUp,location_normalMap,location_normalMapUp;

     GLuint location_texture1,location_texture2;

    TerrainShader(const std::string &multi_file) : MVPTextureShader(multi_file){}
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
    void uploadTexture1(raw_Texture *texture);
    void uploadTexture2(raw_Texture *texture);



};

class Clipmap{
public:
    enum State{
        SW = 0,
        SE = 1,
        NW = 2,
        NE = 3
    };

public:

    static IndexedVertexBuffer<Vertex,GLuint> fixupv,fixuph,degenerated;
     static IndexedVertexBuffer<Vertex,GLuint> trimSW,trimSE,trimNW,trimNE;



    int m;
    vec2 off,scale;

    vec2 cellWidth;
    vec4 offset;
    vec2 ringSize;
//    float f;
    vec2 vp;

    State state;

    bool colored = true;

    Clipmap* next, *previous;

public:


    static TerrainShader* shader;
    static IndexedVertexBuffer<Vertex,GLuint> mesh,center;

    static void createMeshes();





    void init(int m, vec2 off, vec2 scale,State state, Clipmap* next,Clipmap *previous);
    void update(const vec3& p);

    void calculatePosition(vec2 pos);

    void renderRing();
private:
    void render(const IndexedVertexBuffer<Vertex,GLuint> &mesh, vec4 color, vec4 scale,vec4 fineOrigin);
//    void render(const IndexedVertexBuffer<Vertex,GLuint> &mesh, vec4 color, vec4 scale);

    void renderDeg();
    void renderTrim();
    void renderFixUps();
    void renderBlocks();

};
