#include "world/terrain.h"
#include "geometry/triangle_mesh_generator.h"


void TerrainShader::checkUniforms(){
    MVPTextureShader::checkUniforms();
}


Terrain::Terrain(){
    createMesh();
}

void Terrain::createMesh(){


    typedef TriangleMesh<Vertex,GLuint> Mesh;
    Mesh mesh;

    unsigned int w=50;
    unsigned int h = 50;

    float dw = 1.0f/(w-1);
    float dh = 1.0f/(h-1);

    //creating uniform grid with w*h vertices
    //the resulting mesh will fill the quad (0,0,0) - (1,0,1)
    for(unsigned int y=0;y<h;y++){
        for(unsigned int x=0;x<w;x++){
            float fx = (float)x*dw;
            float fy = (float)y*dh;
            Vertex v(vec3(fx,0.0f,fy));
            mesh.addVertex(v);
        }
    }
    for(unsigned int y=0;y<h-1;y++){
        for(unsigned int x=0;x<w-1;x++){
//            GLuint quad[] = {y*w+x,y*w+x+1,(y+1)*w+x+1,(y+1)*w+x};
            GLuint quad[] = {y*w+x,(y+1)*w+x,(y+1)*w+x+1,y*w+x+1};
            mesh.addQuad(quad);
        }
    }

    mesh.createBuffers(this->mesh);

}

void Terrain::setPosition(const vec3& p){
    model[3] = vec4(p,1);
}

void Terrain::setDistance(float d){
    model[0][0] = d;
    model[1][1] = 1;
    model[2][2] = d;
}



void Terrain::render(const mat4& view, const mat4 &proj){
    shader->bind();
//    model[3] = -glm::transpose(view)[3];
    shader->uploadAll(model,view,proj);
    shader->uploadTexture(texture);
    mesh.bindAndDraw();


    shader->unbind();
}
