#include "world/clipmap.h"

IndexedVertexBuffer<Vertex,GLuint> Clipmap::mesh;
IndexedVertexBuffer<Vertex,GLuint> Clipmap::center;
IndexedVertexBuffer<Vertex,GLuint> Clipmap::fixupv;
IndexedVertexBuffer<Vertex,GLuint> Clipmap::fixuph;
IndexedVertexBuffer<Vertex,GLuint> Clipmap::trimSW;
IndexedVertexBuffer<Vertex,GLuint> Clipmap::trimSE;
IndexedVertexBuffer<Vertex,GLuint> Clipmap::trimNW;
IndexedVertexBuffer<Vertex,GLuint> Clipmap::trimNE;
IndexedVertexBuffer<Vertex,GLuint> Clipmap::degenerated;
TerrainShader* Clipmap::shader;




void TerrainShader::checkUniforms(){
    MVPTextureShader::checkUniforms();

    location_ScaleFactor = getUniformLocation("ScaleFactor");
    location_FineBlockOrig = getUniformLocation("FineBlockOrig");
    location_color = getUniformLocation("color");
    location_TexSizeScale = getUniformLocation("TexSizeScale");

    location_RingSize = getUniformLocation("RingSize");
    location_ViewerPos = getUniformLocation("ViewerPos");
    location_AlphaOffset = getUniformLocation("AlphaOffset");
    location_OneOverWidth = getUniformLocation("OneOverWidth");

    location_ZScaleFactor = getUniformLocation("ZScaleFactor");
    location_ZTexScaleFactor = getUniformLocation("ZTexScaleFactor");

    location_normalMap = getUniformLocation("normalMap");
    location_imageUp = getUniformLocation("imageUp");
    location_normalMapUp = getUniformLocation("normalMapUp");

    location_texture1 = getUniformLocation("texture1");
    location_texture2 = getUniformLocation("texture2");
}


void TerrainShader::uploadVP(const vec2 &pos){
    Shader::upload(location_ViewerPos,pos);
}

void TerrainShader::uploadScale(const vec4 &s){
    Shader::upload(location_ScaleFactor,s);
}

void TerrainShader::uploadFineOrigin(const vec4 &s){
    Shader::upload(location_FineBlockOrig,s);
}

void TerrainShader::uploadColor(const vec4 &s){
    Shader::upload(location_color,s);
}

void TerrainShader::uploadTexSizeScale(const vec4 &s){
    Shader::upload(location_TexSizeScale,s);
}

void TerrainShader::uploadRingSize(const vec2 &s){
    Shader::upload(location_RingSize,s);
}

void TerrainShader::uploadZScale(float f){
    Shader::upload(location_ZScaleFactor,f);
}

void TerrainShader::uploadNormalMap(raw_Texture *texture){
    texture->bind(1);
    Shader::upload(location_normalMap,1);
}

void TerrainShader::uploadImageUp(raw_Texture *texture){
    texture->bind(2);
    Shader::upload(location_imageUp,2);
}

void TerrainShader::uploadNormalMapUp(raw_Texture *texture){
    texture->bind(3);
    Shader::upload(location_normalMapUp,3);
}

void TerrainShader::uploadTexture1(raw_Texture *texture){
    texture->bind(4);
    Shader::upload(location_texture1,4);
}

void TerrainShader::uploadTexture2(raw_Texture *texture){
    texture->bind(5);
    Shader::upload(location_texture2,5);
}




void Clipmap::createMeshes()
{

    TerrainMesh tm;

    auto block = tm.createMesh2();
    block->createBuffers(Clipmap::mesh);

    auto fixupv = tm.createMeshFixUpV();
    fixupv->createBuffers(Clipmap::fixupv);

    auto fixuph = tm.createMeshFixUpH();
    fixuph->createBuffers(Clipmap::fixuph);

    auto trim0 = tm.createMeshTrimSW();
    trim0->createBuffers(Clipmap::trimSW);

    auto trim1 = tm.createMeshTrimNE();
    trim1->createBuffers(Clipmap::trimNE);


    auto trim2 = tm.createMeshTrimSE();
    trim2->createBuffers(Clipmap::trimSE);

    auto trim3 = tm.createMeshTrimNW();
    trim3->createBuffers(Clipmap::trimNW);

    auto center = tm.createMeshCenter();
    center->createBuffers(Clipmap::center);

    auto degenerated = tm.createMeshDegenerated();
    degenerated->createBuffers(Clipmap::degenerated);
}

void Clipmap::init(int m, vec2 off, vec2 scale, State state, Clipmap *next, Clipmap *previous)
{
    this->m = m;
    this->off = off;
    this->scale = scale;
    this->state = state;
    this->next = next;
    this->previous = previous;

    cellWidth = scale * (1.0f/(m-1));


    ringSize = 4.0f*scale+2.0f*cellWidth;
}

void Clipmap::update(const vec3 &p)
{
    //round to a multiple of cellwidth


    vp = vec2(p.x,p.z);
//    vp.x = 33.5f;
//    vp.y = 13.1f;


    vp = vp/cellWidth;
    vec2 test = glm::floor(vp);

    vp = test*cellWidth;



}


void Clipmap::calculatePosition(vec2 pos)
{

    vec2 noff(0);
    vec2 relPos(0);

    if(next){
        if(next->vp != vp){
            noff = next->vp - vp;

            state = SE;
        } else{

            state = NE;
        }

    }

    relPos.x = (noff.x==0)?0:1;
    relPos.y = (noff.y==0)?0:1;



    static const vec2 dir[] = {
        vec2(1,1),
        vec2(1,0),
        vec2(0,1),
        vec2(0,0)
    };

    for(int i=0;i<4;i++){
        if(relPos==dir[i])
            state = (State)i;
    }

    vec2 d = dir[state]*cellWidth;
//    vec2 d = dir[state]*cellWidth;


    pos += d;
    off = pos;

    //    offset = vec4(scale.x,scale.y,-scale.x*1.5f-cellWidth.x,-scale.y*1.5f-cellWidth.y);
    offset = vec4(scale.x,scale.y,-scale.x*1.5f,-scale.y*1.5f);
    offset -= vec4(0,0,off.x,off.y);

    pos += noff;

    if(next)
        next->calculatePosition(pos);
}



void Clipmap::renderRing(){



    shader->uploadRingSize(ringSize-cellWidth);
    shader->uploadVP(vp);

    //render 12 blocks
    renderBlocks();

    //render 4 fix up rectangles
    renderFixUps();

    //render L shaped trim
    renderTrim();

    //render degenerated triangles
    renderDeg();
}


void Clipmap::renderDeg(){
    //blockOffset.xz: x offset scaled by 'scale'
    //blockOffset.yw: y offset scaled by 'cellWidth'
    static const vec4 blockOffset[] = {
        vec4(0,0,0,0),
        vec4(2,2,2,2)
    };

    vec2 blockSizeRel = vec2(offset.x/ringSize.x,offset.y/ringSize.y);
    vec2 cellSizeRel = vec2(cellWidth.x/ringSize.x,cellWidth.y/ringSize.y);

    int i = 0;
    vec4 c = vec4(0,1,0,0);
    vec4 s = offset+vec4(0,0,
                         blockOffset[i].x*scale.x+blockOffset[i].y*cellWidth.x,
                         blockOffset[i].z*scale.y+blockOffset[i].w*cellWidth.y);
    vec4 fo = vec4(blockSizeRel.x,
                   blockSizeRel.y,
                   blockOffset[i].x*blockSizeRel.x+blockOffset[i].y*cellSizeRel.x,
                   blockOffset[i].z*blockSizeRel.y+blockOffset[i].w*cellSizeRel.y);



    render(Clipmap::degenerated,c,s,fo);

}

void Clipmap::renderTrim(){
    //blockOffset.xz: x offset scaled by 'scale'
    //blockOffset.yw: y offset scaled by 'cellWidth'
    static const vec4 blockOffset[] = {
        vec4(1,0,1,0),
        vec4(1,0,2,2),
        vec4(2,2,1,0),
        vec4(2,2,2,2)
    };

    vec2 blockSizeRel = vec2(offset.x/ringSize.x,offset.y/ringSize.y);
    vec2 cellSizeRel = vec2(cellWidth.x/ringSize.x,cellWidth.y/ringSize.y);



    vec4 c = vec4(1,0,1,0);
    vec4 s = offset+vec4(0,0,
                         blockOffset[state].x*scale.x+blockOffset[state].y*cellWidth.x,
                         blockOffset[state].z*scale.y+blockOffset[state].w*cellWidth.y);
    vec4 fo = vec4(blockSizeRel.x,
                   blockSizeRel.y,
                   blockOffset[state].x*blockSizeRel.x+blockOffset[state].y*cellSizeRel.x,
                   blockOffset[state].z*blockSizeRel.y+blockOffset[state].w*cellSizeRel.y);

    switch(state){
    case SW:
        render(Clipmap::trimSW,c,s,fo);
        break;
    case SE:
        render(Clipmap::trimSE,c,s,fo);
        break;
    case NW:
        render(Clipmap::trimNW,c,s,fo);
        break;
    case NE:
        render(Clipmap::trimNE,c,s,fo);
        break;
    }

}

void Clipmap::renderFixUps(){
    //blockOffset.xz: x offset scaled by 'scale'
    //blockOffset.yw: y offset scaled by 'cellWidth'
    static const vec4 blockOffset[] = {
        vec4(2,0,0,0),
        vec4(2,0,3,2),
        vec4(0,0,2,0),
        vec4(3,2,2,0)
    };

    vec2 blockSizeRel = vec2(offset.x/ringSize.x,offset.y/ringSize.y);
    vec2 cellSizeRel = vec2(cellWidth.x/ringSize.x,cellWidth.y/ringSize.y);

    for(int i=0;i<4;i++){
        vec4 c = vec4(0,0,1,0);
        vec4 s = offset+vec4(0,0,
                             blockOffset[i].x*scale.x+blockOffset[i].y*cellWidth.x,
                             blockOffset[i].z*scale.y+blockOffset[i].w*cellWidth.y);
        vec4 fo = vec4(blockSizeRel.x,
                       blockSizeRel.y,
                       blockOffset[i].x*blockSizeRel.x+blockOffset[i].y*cellSizeRel.x,
                       blockOffset[i].z*blockSizeRel.y+blockOffset[i].w*cellSizeRel.y);

        if(i>=2)
            render(Clipmap::fixuph,c,s,fo);
        else

            render(Clipmap::fixupv,c,s,fo);
    }

}

void Clipmap::renderBlocks(){

    //blockOffset.xz: x offset scaled by 'scale'
    //blockOffset.yw: y offset scaled by 'cellWidth'
    static const vec4 blockOffset[] = {
        vec4(0,0,0,0), //topleft
        vec4(1,0,0,0),
        vec4(2,2,0,0),
        vec4(3,2,0,0), //topright

        vec4(0,0,1,0),
        vec4(3,2,1,0),
        vec4(0,0,2,2),
        vec4(3,2,2,2),

        vec4(0,0,3,2), //bottomleft
        vec4(1,0,3,2),
        vec4(2,2,3,2),
        vec4(3,2,3,2)   //bottomright
    };

    vec2 blockSizeRel = vec2(offset.x/ringSize.x,offset.y/ringSize.y);
    vec2 cellSizeRel = vec2(cellWidth.x/ringSize.x,cellWidth.y/ringSize.y);

    for(int i=0;i<12;i++){
        vec4 c = vec4(1,0,0,0);
        vec4 s = offset+vec4(0,0,
                             blockOffset[i].x*scale.x+blockOffset[i].y*cellWidth.x,
                             blockOffset[i].z*scale.y+blockOffset[i].w*cellWidth.y);
        vec4 fo = vec4(blockSizeRel.x,
                       blockSizeRel.y,
                       blockOffset[i].x*blockSizeRel.x+blockOffset[i].y*cellSizeRel.x,
                       blockOffset[i].z*blockSizeRel.y+blockOffset[i].w*cellSizeRel.y);

        render(Clipmap::mesh,c,s,fo);
    }
}

void Clipmap::render(const IndexedVertexBuffer<Vertex,GLuint> &mesh, vec4 color,vec4 scale,vec4 fineOrigin){
    shader->uploadScale(scale);

    if(colored)
        shader->uploadColor(color);

    //    shader->uploadFineOrigin(fineOrigin);
    mesh.bindAndDraw();
}

//void Clipmap::render(const IndexedVertexBuffer<Vertex,GLuint> &mesh, vec4 color,vec4 scale){
//    shader->uploadScale(scale);
//    shader->uploadColor(color);
//    mesh.bindAndDraw();
//}
