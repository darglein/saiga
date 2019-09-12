/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/world/clipmap.h"

namespace Saiga
{
IndexedVertexBuffer<Vertex, GLuint> Clipmap::mesh;
IndexedVertexBuffer<Vertex, GLuint> Clipmap::center;
IndexedVertexBuffer<Vertex, GLuint> Clipmap::fixupv;
IndexedVertexBuffer<Vertex, GLuint> Clipmap::fixuph;
IndexedVertexBuffer<Vertex, GLuint> Clipmap::trimSW;
IndexedVertexBuffer<Vertex, GLuint> Clipmap::trimSE;
IndexedVertexBuffer<Vertex, GLuint> Clipmap::trimNW;
IndexedVertexBuffer<Vertex, GLuint> Clipmap::trimNE;
IndexedVertexBuffer<Vertex, GLuint> Clipmap::degenerated;
std::shared_ptr<TerrainShader> Clipmap::shader;



void TerrainShader::checkUniforms()
{
    MVPTextureShader::checkUniforms();

    location_ScaleFactor   = getUniformLocation("ScaleFactor");
    location_FineBlockOrig = getUniformLocation("FineBlockOrig");
    location_color         = getUniformLocation("color");
    location_TexSizeScale  = getUniformLocation("TexSizeScale");

    location_RingSize     = getUniformLocation("RingSize");
    location_ViewerPos    = getUniformLocation("ViewerPos");
    location_AlphaOffset  = getUniformLocation("AlphaOffset");
    location_OneOverWidth = getUniformLocation("OneOverWidth");

    location_ZScaleFactor    = getUniformLocation("ZScaleFactor");
    location_ZTexScaleFactor = getUniformLocation("ZTexScaleFactor");

    location_normalMap   = getUniformLocation("normalMap");
    location_imageUp     = getUniformLocation("imageUp");
    location_normalMapUp = getUniformLocation("normalMapUp");

    location_texture1 = getUniformLocation("texture1");
    location_texture2 = getUniformLocation("texture2");
}


void TerrainShader::uploadVP(const vec2& pos)
{
    Shader::upload(location_ViewerPos, pos);
}

void TerrainShader::uploadScale(const vec4& s)
{
    Shader::upload(location_ScaleFactor, s);
}

void TerrainShader::uploadFineOrigin(const vec4& s)
{
    Shader::upload(location_FineBlockOrig, s);
}

void TerrainShader::uploadColor(const vec4& s)
{
    Shader::upload(location_color, s);
}

void TerrainShader::uploadTexSizeScale(const vec4& s)
{
    Shader::upload(location_TexSizeScale, s);
}

void TerrainShader::uploadRingSize(const vec2& s)
{
    Shader::upload(location_RingSize, s);
}

void TerrainShader::uploadZScale(float f)
{
    Shader::upload(location_ZScaleFactor, f);
}

void TerrainShader::uploadNormalMap(std::shared_ptr<TextureBase> texture)
{
    texture->bind(1);
    Shader::upload(location_normalMap, 1);
}

void TerrainShader::uploadImageUp(std::shared_ptr<TextureBase> texture)
{
    texture->bind(2);
    Shader::upload(location_imageUp, 2);
}

void TerrainShader::uploadNormalMapUp(std::shared_ptr<TextureBase> texture)
{
    texture->bind(3);
    Shader::upload(location_normalMapUp, 3);
}

void TerrainShader::uploadTexture1(std::shared_ptr<TextureBase> texture)
{
    texture->bind(4);
    Shader::upload(location_texture1, 4);
}

void TerrainShader::uploadTexture2(std::shared_ptr<TextureBase> texture)
{
    texture->bind(5);
    Shader::upload(location_texture2, 5);
}



void Clipmap::createMeshes()
{
    TerrainMesh tm;

    //    auto block = tm.createMesh2();
    //    block->createBuffers(Clipmap::mesh);

    //    auto fixupv = tm.createMeshFixUpV();
    //    fixupv->createBuffers(Clipmap::fixupv);

    //    auto fixuph = tm.createMeshFixUpH();
    //    fixuph->createBuffers(Clipmap::fixuph);

    //    auto trim0 = tm.createMeshTrimSW();
    //    trim0->createBuffers(Clipmap::trimSW);

    //    auto trim1 = tm.createMeshTrimNE();
    //    trim1->createBuffers(Clipmap::trimNE);


    //    auto trim2 = tm.createMeshTrimSE();
    //    trim2->createBuffers(Clipmap::trimSE);

    //    auto trim3 = tm.createMeshTrimNW();
    //    trim3->createBuffers(Clipmap::trimNW);

    //    auto center = tm.createMeshCenter();
    //    center->createBuffers(Clipmap::center);

    //    auto degenerated = tm.createMeshDegenerated();
    //    degenerated->createBuffers(Clipmap::degenerated);
}

void Clipmap::init(int m, vec2 off, vec2 scale, State state, Clipmap* next, Clipmap* previous)
{
    this->m        = m;
    this->off      = off;
    this->scale    = scale;
    this->state    = state;
    this->next     = next;
    this->previous = previous;

    cellWidth = scale * (1.0f / (m - 1));


    ringSize = 4.0f * scale + 2.0f * cellWidth;
}

void Clipmap::update(const vec3& p)
{
#if 0
    // round to a multiple of cellwidth


    vp = vec2(p[0], p[2]);
    //    vp[0] = 33.5f;
    //    vp[1] = 13.1f;


    vp        = ele_div(vp, cellWidth);
    vec2 test = floor(vp);

    vp = test * cellWidth;
#endif
}


void Clipmap::calculatePosition(vec2 pos)
{
#if 0
    vec2 noff(0);
    vec2 relPos(0);

    if (next)
    {
        if (next->vp != vp)
        {
            noff = next->vp - vp;

            state = SE;
        }
        else
        {
            state = NE;
        }
    }

    relPos[0] = (noff[0] == 0) ? 0 : 1;
    relPos[1] = (noff[1] == 0) ? 0 : 1;



    static const vec2 dir[] = {vec2(1, 1), vec2(1, 0), vec2(0, 1), vec2(0, 0)};

    for (int i = 0; i < 4; i++)
    {
        if (relPos == dir[i]) state = (State)i;
    }

    vec2 d = dir[state] * cellWidth;
    //    vec2 d = dir[state]*cellWidth;


    pos += d;
    off = pos;

    //    offset = vec4(scale[0],scale[1],-scale[0]*1.5f-cellWidth[0],-scale[1]*1.5f-cellWidth[1]);
    offset = vec4(scale[0], scale[1], -scale[0] * 1.5f, -scale[1] * 1.5f);
    offset -= vec4(0, 0, off[0], off[1]);

    pos += noff;

    if (next) next->calculatePosition(pos);
#endif
}



void Clipmap::renderRing()
{
    shader->uploadRingSize(ringSize - cellWidth);
    shader->uploadVP(vp);

    // render 12 blocks
    renderBlocks();

    // render 4 fix up rectangles
    renderFixUps();

    // render L shaped trim
    renderTrim();

    // render degenerated triangles
    renderDeg();
}


void Clipmap::renderDeg()
{
    // blockOffset[0]z: x offset scaled by 'scale'
    // blockOffset[1]w: y offset scaled by 'cellWidth'
    static const vec4 blockOffset[] = {vec4(0, 0, 0, 0), vec4(2, 2, 2, 2)};

    vec2 blockSizeRel = vec2(offset[0] / ringSize[0], offset[1] / ringSize[1]);
    vec2 cellSizeRel  = vec2(cellWidth[0] / ringSize[0], cellWidth[1] / ringSize[1]);

    int i  = 0;
    vec4 c = vec4(0, 1, 0, 0);
    vec4 s = offset + vec4(0, 0, blockOffset[i][0] * scale[0] + blockOffset[i][1] * cellWidth[0],
                           blockOffset[i][2] * scale[1] + blockOffset[i][3] * cellWidth[1]);
    vec4 fo =
        vec4(blockSizeRel[0], blockSizeRel[1], blockOffset[i][0] * blockSizeRel[0] + blockOffset[i][1] * cellSizeRel[0],
             blockOffset[i][2] * blockSizeRel[1] + blockOffset[i][3] * cellSizeRel[1]);



    render(Clipmap::degenerated, c, s, fo);
}

void Clipmap::renderTrim()
{
    // blockOffset[0]z: x offset scaled by 'scale'
    // blockOffset[1]w: y offset scaled by 'cellWidth'
    static const vec4 blockOffset[] = {vec4(1, 0, 1, 0), vec4(1, 0, 2, 2), vec4(2, 2, 1, 0), vec4(2, 2, 2, 2)};

    vec2 blockSizeRel = vec2(offset[0] / ringSize[0], offset[1] / ringSize[1]);
    vec2 cellSizeRel  = vec2(cellWidth[0] / ringSize[0], cellWidth[1] / ringSize[1]);



    vec4 c  = vec4(1, 0, 1, 0);
    vec4 s  = offset + vec4(0, 0, blockOffset[state][0] * scale[0] + blockOffset[state][1] * cellWidth[0],
                           blockOffset[state][2] * scale[1] + blockOffset[state][3] * cellWidth[1]);
    vec4 fo = vec4(blockSizeRel[0], blockSizeRel[1],
                   blockOffset[state][0] * blockSizeRel[0] + blockOffset[state][1] * cellSizeRel[0],
                   blockOffset[state][2] * blockSizeRel[1] + blockOffset[state][3] * cellSizeRel[1]);

    switch (state)
    {
        case SW:
            render(Clipmap::trimSW, c, s, fo);
            break;
        case SE:
            render(Clipmap::trimSE, c, s, fo);
            break;
        case NW:
            render(Clipmap::trimNW, c, s, fo);
            break;
        case NE:
            render(Clipmap::trimNE, c, s, fo);
            break;
    }
}

void Clipmap::renderFixUps()
{
    // blockOffset[0]z: x offset scaled by 'scale'
    // blockOffset[1]w: y offset scaled by 'cellWidth'
    static const vec4 blockOffset[] = {vec4(2, 0, 0, 0), vec4(2, 0, 3, 2), vec4(0, 0, 2, 0), vec4(3, 2, 2, 0)};

    vec2 blockSizeRel = vec2(offset[0] / ringSize[0], offset[1] / ringSize[1]);
    vec2 cellSizeRel  = vec2(cellWidth[0] / ringSize[0], cellWidth[1] / ringSize[1]);

    for (int i = 0; i < 4; i++)
    {
        vec4 c  = vec4(0, 0, 1, 0);
        vec4 s  = offset + vec4(0, 0, blockOffset[i][0] * scale[0] + blockOffset[i][1] * cellWidth[0],
                               blockOffset[i][2] * scale[1] + blockOffset[i][3] * cellWidth[1]);
        vec4 fo = vec4(blockSizeRel[0], blockSizeRel[1],
                       blockOffset[i][0] * blockSizeRel[0] + blockOffset[i][1] * cellSizeRel[0],
                       blockOffset[i][2] * blockSizeRel[1] + blockOffset[i][3] * cellSizeRel[1]);

        if (i >= 2)
            render(Clipmap::fixuph, c, s, fo);
        else

            render(Clipmap::fixupv, c, s, fo);
    }
}

void Clipmap::renderBlocks()
{
    // blockOffset[0]z: x offset scaled by 'scale'
    // blockOffset[1]w: y offset scaled by 'cellWidth'
    static const vec4 blockOffset[] = {
        vec4(0, 0, 0, 0),                                      // topleft
        vec4(1, 0, 0, 0), vec4(2, 2, 0, 0), vec4(3, 2, 0, 0),  // topright

        vec4(0, 0, 1, 0), vec4(3, 2, 1, 0), vec4(0, 0, 2, 2), vec4(3, 2, 2, 2),

        vec4(0, 0, 3, 2),                                     // bottomleft
        vec4(1, 0, 3, 2), vec4(2, 2, 3, 2), vec4(3, 2, 3, 2)  // bottomright
    };

    vec2 blockSizeRel = vec2(offset[0] / ringSize[0], offset[1] / ringSize[1]);
    vec2 cellSizeRel  = vec2(cellWidth[0] / ringSize[0], cellWidth[1] / ringSize[1]);

    for (int i = 0; i < 12; i++)
    {
        vec4 c  = vec4(1, 0, 0, 0);
        vec4 s  = offset + vec4(0, 0, blockOffset[i][0] * scale[0] + blockOffset[i][1] * cellWidth[0],
                               blockOffset[i][2] * scale[1] + blockOffset[i][3] * cellWidth[1]);
        vec4 fo = vec4(blockSizeRel[0], blockSizeRel[1],
                       blockOffset[i][0] * blockSizeRel[0] + blockOffset[i][1] * cellSizeRel[0],
                       blockOffset[i][2] * blockSizeRel[1] + blockOffset[i][3] * cellSizeRel[1]);

        render(Clipmap::mesh, c, s, fo);
    }
}

void Clipmap::render(const IndexedVertexBuffer<Vertex, GLuint>& mesh, vec4 color, vec4 scale, vec4 fineOrigin)
{
    shader->uploadScale(scale);

    if (colored) shader->uploadColor(color);

    //    shader->uploadFineOrigin(fineOrigin);
    mesh.bindAndDraw();
}

// void Clipmap::render(const IndexedVertexBuffer<Vertex,GLuint> &mesh, vec4 color,vec4 scale){
//    shader->uploadScale(scale);
//    shader->uploadColor(color);
//    mesh.bindAndDraw();
//}

}  // namespace Saiga
