#include "world/terrain.h"



void TerrainShader::checkUniforms(){
    MVPTextureShader::checkUniforms();

    location_ScaleFactor = getUniformLocation("ScaleFactor");
    location_FineBlockOrig = getUniformLocation("FineBlockOrig");
    location_color = getUniformLocation("color");

    location_ViewerPos = getUniformLocation("ViewerPos");
    location_AlphaOffset = getUniformLocation("AlphaOffset");
    location_OneOverWidth = getUniformLocation("OneOverWidth");

    location_ZScaleFactor = getUniformLocation("ZScaleFactor");
    location_ZTexScaleFactor = getUniformLocation("ZTexScaleFactor");
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

Terrain::Terrain():heightmap(1000,1000){

}

void Terrain::createMesh(unsigned int w, unsigned int h){



    auto block = heightmap.createMesh2();
    block->createBuffers(this->mesh);

    auto fixupv = heightmap.createMeshFixUpV();
    fixupv->createBuffers(this->fixupv);

    auto fixuph = heightmap.createMeshFixUpH();
    fixuph->createBuffers(this->fixuph);

    auto trim = heightmap.createMeshTrim();
    trim->createBuffers(this->trim);

    auto trimi = heightmap.createMeshTrimi();
    trimi->createBuffers(this->trimi);

    auto center = heightmap.createMeshCenter();
    center->createBuffers(this->center);

    heightmap.createTestHeightmap();
    this->texture = heightmap.createTexture();

}

void Terrain::setPosition(const vec3& p){
    model[3] = vec4(p,1);
}

void Terrain::setDistance(float d){
    model[0][0] = d;
    model[1][1] = 1;
    model[2][2] = d;
}



void Terrain::render(const vec3 &viewPos, const mat4& view, const mat4 &proj){
    this->viewPos = viewPos;

    shader->bind();

    shader->uploadAll(model,view,proj);
    shader->uploadTexture(texture);
    vec2 vp(viewPos.x,viewPos.z);
    shader->uploadVP(vp);

    //    renderBlocks(vec2(10,10),1);
    renderCenter(vec4(1,1,0,0),vec2(40,40));


    vec2 baseScale(20,20);
    vec2 baseCellWidth = baseScale * (1.0f/(heightmap.m-1));

    //    vec2 offsets[] = {vec2(0),-baseCellWidth,baseCellWidth,-3.0f*baseCellWidth,5.0f*baseCellWidth};
    float offsets[] = {0,-1,1,-3,5,-11,21};

    vec2 scale = baseScale;
    for(int i=0;i<7;i++){
        renderRing(scale,(i+1)%2,offsets[i]*baseCellWidth);
        scale*=2.0f;
    }


    shader->unbind();
}

void Terrain::renderRing(vec2 scale, float f, vec2 off){
    vec2 cellWidth = scale * (1.0f/(heightmap.m-1));
    vec4 offset = vec4(scale.x,scale.y,-scale.x*1.5f-cellWidth.x,-scale.y*1.5f-cellWidth.y);
    offset -= vec4(0,0,off.x,off.y);

    //render 12 blocks
    renderBlocks(scale,cellWidth,offset);

    //render 4 fix up rectangles
    renderFixUpV(vec4(0,0,1,0),offset+vec4(0,0,scale.x*2,0));
    renderFixUpV(vec4(0,0,1,0),offset+vec4(0,0,scale.x*2,scale.y*3+cellWidth.y*2));
    renderFixUpH(vec4(0,0,1,0),offset+vec4(0,0,0,scale.y*2));
    renderFixUpH(vec4(0,0,1,0),offset+vec4(0,0,scale.x*3+cellWidth.x*2,scale.y*2));

    //render L shaped trim
    if(f>0)
        renderTrim(vec4(1,1,1,0),offset+vec4(0,0,scale.x,scale.y));
    else
        renderTrimi(vec4(1,1,1,0),offset+vec4(0,0,scale.x*2+cellWidth.x*2,scale.y*2+cellWidth.y*2));
}


void Terrain::renderBlocks(vec2 scale,vec2 cellWidth, vec4 offset){

    //blockOffset.xz: x offset scaled by 'scale'
    //blockOffset.yw: y offset scaled by 'cellWidth'
    const vec4 blockOffset[] = {
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

    for(int i=0;i<12;i++){
        renderBlock(vec4(1,0,0,0),offset+vec4(0,0,
                                              blockOffset[i].x*scale.x+blockOffset[i].y*cellWidth.x,
                                              blockOffset[i].z*scale.y+blockOffset[i].w*cellWidth.y));
    }
}

void Terrain::renderBlock(vec4 color,vec4 scale){


    shader->uploadScale(scale);
    shader->uploadColor(color);
    mesh.bindAndDraw();
}

void Terrain::renderFixUpV(vec4 color,vec4 scale){
    shader->uploadScale(scale);
    shader->uploadColor(color);
    fixupv.bindAndDraw();
}

void Terrain::renderFixUpH(vec4 color,vec4 scale){
    shader->uploadScale(scale);
    shader->uploadColor(color);
    fixuph.bindAndDraw();
}

void Terrain::renderTrim(vec4 color,vec4 scale){
    shader->uploadScale(scale);
    shader->uploadColor(color);
    trim.bindAndDraw();
}

void Terrain::renderTrimi(vec4 color,vec4 scale){
    shader->uploadScale(scale);
    shader->uploadColor(color);
    trimi.bindAndDraw();
}

void Terrain::renderCenter(vec4 color,vec2 scale){
    shader->uploadScale(vec4(scale.x,scale.y,0,0));
    shader->uploadColor(color);
    center.bindAndDraw();
}


