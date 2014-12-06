#include "world/terrain.h"



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


Terrain::Terrain():heightmap(7,1024,1024){ //1024,2560,4096

}

void Terrain::createMesh(unsigned int w, unsigned int h){


    TerrainMesh tm;

    auto block = tm.createMesh2();
    block->createBuffers(this->mesh);

    auto fixupv = tm.createMeshFixUpV();
    fixupv->createBuffers(this->fixupv);

    auto fixuph = tm.createMeshFixUpH();
    fixuph->createBuffers(this->fixuph);

    auto trim = tm.createMeshTrim();
    trim->createBuffers(this->trim);

    auto trimi = tm.createMeshTrimi();
    trimi->createBuffers(this->trimi);

    auto center = tm.createMeshCenter();
    center->createBuffers(this->center);

    auto degenerated = tm.createMeshDegenerated();
    degenerated->createBuffers(this->degenerated);


    heightmap.createHeightmaps();

   heightmap.createTextures();

}

void Terrain::setPosition(const vec3& p){
    model[3] = vec4(p,1);
}

void Terrain::setDistance(float d){
    model[0][0] = d;
    model[1][1] = 1;
    model[2][2] = d;
}



void Terrain::render(Camera *cam){
    shader = deferredshader;
    renderintern(cam);
}

void Terrain::renderDepth(Camera* cam){
    shader = depthshader;
    renderintern(cam);
}

void Terrain::renderintern(Camera *cam){
//    this->viewPos = cam->m;
//    this->viewPos = vec3(0);

    shader->bind();

    shader->uploadAll(model,cam->view,cam->proj);
    vec2 vp(this->viewPos.x,this->viewPos.z);
    shader->uploadVP(vp);
    shader->uploadZScale(heightmap.heightScale);
    shader->uploadTexSizeScale(vec4(heightmap.w,heightmap.h,heightmap.w/2000.0f,heightmap.h/2000.0f));

    //    renderBlocks(vec2(10,10),1);
    shader->uploadTexture(heightmap.texheightmap[0]);
    shader->uploadNormalMap(heightmap.texnormalmap[0]);
    shader->uploadImageUp(heightmap.texheightmap[0]);
    shader->uploadNormalMapUp(heightmap.texnormalmap[0]);
    render(center,vec4(1,1,0,0),vec4(40,40,0,0),vec4(1,1,0,0));


    vec2 baseScale(20,20);
    vec2 baseCellWidth = baseScale * (1.0f/(heightmap.m-1));

    //    vec2 offsets[] = {vec2(0),-baseCellWidth,baseCellWidth,-3.0f*baseCellWidth,5.0f*baseCellWidth};
    float offsets[] = {0,-1,1,-3,5,-11,21};

    vec2 scale = baseScale;
    for(int i=0;i<7;i++){
        shader->uploadTexture(heightmap.texheightmap[i]);
        shader->uploadNormalMap(heightmap.texnormalmap[0]);

        int next = glm::clamp(i+1,0,6);
        shader->uploadImageUp(heightmap.texheightmap[next]);
        shader->uploadNormalMapUp(heightmap.texnormalmap[0]);
        renderRing(scale,(i+1)%2,offsets[i]*baseCellWidth);
        scale*=2.0f;
    }


    shader->unbind();
}

void Terrain::renderRing(vec2 scale, float f, vec2 off){
    vec2 cellWidth = scale * (1.0f/(heightmap.m-1));
    vec4 offset = vec4(scale.x,scale.y,-scale.x*1.5f-cellWidth.x,-scale.y*1.5f-cellWidth.y);
    offset -= vec4(0,0,off.x,off.y);

    vec2 ringSize = 4.0f*scale+2.0f*cellWidth;

    shader->uploadRingSize(ringSize-cellWidth);

    //render 12 blocks
    renderBlocks(scale,cellWidth,offset,ringSize);

    //render 4 fix up rectangles
    renderFixUps(scale,cellWidth,offset,ringSize);

    //render L shaped trim
    renderTrim(scale,cellWidth,offset,ringSize,f);

    //render degenerated triangles
    renderDeg(scale,cellWidth,offset,ringSize);
}


void Terrain::renderDeg(vec2 scale,vec2 cellWidth, vec4 offset,vec2 ringSize){
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



        render(degenerated,c,s,fo);

}

void Terrain::renderTrim(vec2 scale,vec2 cellWidth, vec4 offset,vec2 ringSize,float f){
    //blockOffset.xz: x offset scaled by 'scale'
    //blockOffset.yw: y offset scaled by 'cellWidth'
    static const vec4 blockOffset[] = {
        vec4(1,0,1,0),
        vec4(2,2,2,2)
    };

    vec2 blockSizeRel = vec2(offset.x/ringSize.x,offset.y/ringSize.y);
    vec2 cellSizeRel = vec2(cellWidth.x/ringSize.x,cellWidth.y/ringSize.y);

    int i = (f>0)?0:1;
    vec4 c = vec4(1,0,1,0);
    vec4 s = offset+vec4(0,0,
                         blockOffset[i].x*scale.x+blockOffset[i].y*cellWidth.x,
                         blockOffset[i].z*scale.y+blockOffset[i].w*cellWidth.y);
    vec4 fo = vec4(blockSizeRel.x,
                   blockSizeRel.y,
                   blockOffset[i].x*blockSizeRel.x+blockOffset[i].y*cellSizeRel.x,
                   blockOffset[i].z*blockSizeRel.y+blockOffset[i].w*cellSizeRel.y);


    if(i)
        render(trimi,c,s,fo);
    else
        render(trim,c,s,fo);
}

void Terrain::renderFixUps(vec2 scale,vec2 cellWidth, vec4 offset,vec2 ringSize){
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
            render(fixuph,c,s,fo);
        else

            render(fixupv,c,s,fo);
    }

}

void Terrain::renderBlocks(vec2 scale,vec2 cellWidth, vec4 offset,vec2 ringSize){

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

        render(mesh,c,s,fo);
    }
}

void Terrain::render(const IndexedVertexBuffer<Vertex,GLuint> &mesh, vec4 color,vec4 scale,vec4 fineOrigin){
    shader->uploadScale(scale);
    shader->uploadColor(color);
    shader->uploadFineOrigin(fineOrigin);
    mesh.bindAndDraw();
}

void Terrain::render(const IndexedVertexBuffer<Vertex,GLuint> &mesh, vec4 color,vec4 scale){
    shader->uploadScale(scale);
    shader->uploadColor(color);
    mesh.bindAndDraw();
}



