#include "world/terrain.h"



Terrain::Terrain(int layers, int w, int h, float heightScale):layers(layers),heightmap(layers,w,h){ //1024,2048,4096
    clipmaps.resize(layers+1);
    heightmap.heightScale = heightScale;
}

bool Terrain::loadHeightmap(){
    if (!heightmap.loadMaps())
        return false;

    heightmap.createTextures();
    return true;
}

void Terrain::createHeightmap(){
    heightmap.createHeightmaps();
}

void Terrain::createMesh(){


    Clipmap::createMeshes();

//    float offsets[] = {0,-1,1,-3,5,-11,21};
    float offsets[] = {0,0,0,0,0,0,0};
//    Clipmap::State states[] = {Clipmap::NW,Clipmap::NW,Clipmap::NW,Clipmap::NW,Clipmap::NW,Clipmap::NW,Clipmap::NW};
//    Clipmap::State states[] = {Clipmap::NE,Clipmap::NE,Clipmap::NE,Clipmap::NE,Clipmap::NE,Clipmap::NE,Clipmap::NE};
//    Clipmap::State states[] = {Clipmap::SW,Clipmap::SW,Clipmap::SW,Clipmap::SW,Clipmap::SW,Clipmap::SW,Clipmap::SW};
//    Clipmap::State states[] = {Clipmap::SE,Clipmap::SE,Clipmap::SE,Clipmap::SE,Clipmap::SE,Clipmap::SE,Clipmap::SE};

    Clipmap::State states[] = {Clipmap::NE,Clipmap::SW,Clipmap::SE,Clipmap::SE,Clipmap::NW,Clipmap::NE,Clipmap::SW};


    vec2 baseCellWidth = baseScale * (1.0f/(heightmap.m-1));
     vec2 scale = baseScale;

    for(int i=0;i<layers;++i){
        Clipmap* n = (i+1<layers)?&clipmaps[i+1]:nullptr;
        Clipmap* p = (i-1>0)?&clipmaps[i-1]:nullptr;

        clipmaps[i].init(heightmap.m,offsets[i]*baseCellWidth,scale,states[i],n,p);
         scale*=2.0f;
    }


    cout<<"Terrain initialized!"<<endl;

}



void Terrain::setDistance(float d){
    model[0][0] = d;
    model[1][1] = 1;
    model[2][2] = d;
}



void Terrain::render(Camera *cam){
    shader = deferredshader;
    Clipmap::shader = deferredshader;

//    update(vec3(cam->getPosition()));
    renderintern(cam);
}

void Terrain::renderDepth(Camera* cam){
    shader = depthshader;
    Clipmap::shader = depthshader;
//    update(vec3(cam->getPosition()));
    renderintern(cam);
}

void Terrain::update(const vec3 &p)
{
    viewPos = p;
    for(int i=0;i<layers;++i){
        clipmaps[i].update(p);
    }

    //random states (LOL)
//    for(int i=0;i<levels;++i){
//        glm::linearRand(vec3(0),vec3(1));
//        clipmaps[i].state = (Clipmap::State)(rand()%4);
//    }

    clipmaps[0].calculatePosition(vec2(0));
}

void Terrain::renderintern(Camera *cam){
    //    this->viewPos = cam->m;
    //    this->viewPos = vec3(0);

    shader->bind();

    shader->uploadAll(model,cam->view,cam->proj);


    shader->uploadZScale(heightmap.heightScale);

    vec4 TexSizeScale = vec4(heightmap.mapOffset.x,heightmap.mapOffset.y,heightmap.mapScaleInv.x,heightmap.mapScaleInv.y);
    shader->uploadTexSizeScale(TexSizeScale);


    shader->uploadTexture(heightmap.texheightmap[0]);
    shader->uploadNormalMap(heightmap.texnormalmap[0]);
    shader->uploadImageUp(heightmap.texheightmap[0]);
    shader->uploadNormalMapUp(heightmap.texnormalmap[0]);
    shader->uploadColor(vec4(1));

    shader->uploadTexture1(texture1);
    shader->uploadTexture2(texture2);

    shader->uploadVP(clipmaps[0].vp);
    render(Clipmap::center,vec4(1),vec4(baseScale.x*2,baseScale.y*2,0,0),vec4(1,1,0,0));



    for(int i=0;i<layers-1;i++){
        shader->uploadTexture(heightmap.texheightmap[i]);
        shader->uploadNormalMap(heightmap.texnormalmap[0]);

        int next = glm::clamp(i+1,0,layers-2);
        shader->uploadImageUp(heightmap.texheightmap[next]);
        shader->uploadNormalMapUp(heightmap.texnormalmap[0]);

        clipmaps[i].renderRing();

    }


    shader->unbind();
}


void Terrain::render(const IndexedVertexBuffer<Vertex,GLuint> &mesh, vec4 color,vec4 scale,vec4 fineOrigin){
    shader->uploadScale(scale);
    shader->uploadColor(color);
//    shader->uploadFineOrigin(fineOrigin);
    mesh.bindAndDraw();
}



