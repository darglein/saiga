#include "world/terrainmesh.h"

TerrainMesh::TerrainMesh(){


}

std::shared_ptr<TerrainMesh::mesh_t> TerrainMesh::createMesh(){

    unsigned int w = 100;
    unsigned int h = 100;


    return createGridMesh(w,h,vec2(2.0f/(w-1),2.0f/(h-1)),vec2(1.0));

}


std::shared_ptr<TerrainMesh::mesh_t> TerrainMesh::createMesh2(){
    return createGridMesh(m,m,vec2(1.0f/(m-1)),vec2(0.5));
}



std::shared_ptr<TerrainMesh::mesh_t> TerrainMesh::createMeshFixUpV(){
    return createGridMesh(3,m,vec2(1.0f/(m-1)),vec2(0.5));
}



std::shared_ptr<TerrainMesh::mesh_t> TerrainMesh::createMeshFixUpH(){
    return createGridMesh(m,3,vec2(1.0f/(m-1)),vec2(0.5));
}

std::shared_ptr<TerrainMesh::mesh_t> TerrainMesh::createMeshTrimSW(){

    mesh_t* mesh = new mesh_t();

    int w = 2*m+1;
    int h = 2;
    vec2 d = vec2(1.0f/(m-1));
    vec2 o = vec2(0.5);

    float dw = d.x;
    float dh = d.y;
    for(unsigned int y=0;y<h;y++){
        for(unsigned int x=0;x<w;x++){
            float fx = (float)x*dw-o.x;
            float fy = (float)y*dh-o.y;
            Vertex v(vec3(fx,0.0f,fy));
            mesh->addVertex(v);
        }
    }


    for(unsigned int y=0;y<h-1;y++){
        for(unsigned int x=0;x<w-1;x++){
            GLuint quad[] = {y*w+x,(y+1)*w+x,(y+1)*w+x+1,y*w+x+1};
            mesh->addQuad(quad);
        }
    }

    int offset = mesh->vertices.size();

    w = 2;
    h = 2*m;

    dw = d.x;
    dh = d.y;
    for(unsigned int y=0;y<h;y++){
        for(unsigned int x=0;x<w;x++){
            float fx = (float)x*dw-o.x;
            float fy = (float)(y+1)*dh-o.y;
            Vertex v(vec3(fx,0.0f,fy));
            mesh->addVertex(v);
        }
    }


    for(unsigned int y=0;y<h-1;y++){
        for(unsigned int x=0;x<w-1;x++){
            GLuint quad[] = {offset+y*w+x,offset+(y+1)*w+x,offset+(y+1)*w+x+1,offset+y*w+x+1};
            mesh->addQuad(quad);
        }
    }

    return std::shared_ptr<TerrainMesh::mesh_t>(mesh);
}

std::shared_ptr<TerrainMesh::mesh_t> TerrainMesh::createMeshTrimSE(){

    auto mesh = createMeshTrimSW();
    mesh->transform(glm::rotate(mat4(),glm::radians(90.0f),vec3(0,1,0)));
    return mesh;
}

std::shared_ptr<TerrainMesh::mesh_t> TerrainMesh::createMeshTrimNW(){

    auto mesh = createMeshTrimSW();
    mesh->transform(glm::rotate(mat4(),glm::radians(-90.0f),vec3(0,1,0)));
    return mesh;
}


std::shared_ptr<TerrainMesh::mesh_t> TerrainMesh::createMeshTrimNE(){

    auto mesh = createMeshTrimSW();
    mesh->transform(glm::rotate(mat4(),glm::radians(180.0f),vec3(0,1,0)));
    return mesh;
}




std::shared_ptr<TerrainMesh::mesh_t> TerrainMesh::createMeshDegenerated(){
    mesh_t* mesh = new mesh_t();

    int w = (n+1)/2;
    float dx = 2.0f/(m-1);

    vec2 d[] = {vec2(dx,0),vec2(dx,0),vec2(0,dx),vec2(0,dx)};
    vec2 o[] = {vec2(0.5),vec2(0.5,-3.5-(dx)),vec2(0.5),vec2(-3.5-(dx),0.5)};

    int orientation[] = {1,0,0,1};



    for(int i=0;i<4;i++){
        int offset = mesh->vertices.size();

        for(unsigned int x=0;x<w;x++){
            float fx = (float)x*d[i].x-o[i].x;
            float fy = (float)x*d[i].y-o[i].y;
            Vertex v(vec3(fx,0.0f,fy));
            mesh->addVertex(v);
            if(x<w-1){
                //add vertex between
                fx = fx+0.5f*d[i].x;
                fy = fy+0.5f*d[i].y;
                Vertex v(vec3(fx,0.0f,fy));
                mesh->addVertex(v);
            }

        }


        for(unsigned int x=0;x<w-1;x++){
            //add degenerated triangle
            unsigned int idx = 2*x;
            GLuint face1[] = {offset+idx,offset+idx+1,offset+idx+2};
            GLuint face2[] = {offset+idx,offset+idx+2,offset+idx+1};

//            if(orientation[i])
                mesh->addFace(face1);
//            else
                mesh->addFace(face2);
        }


    }




    return std::shared_ptr<TerrainMesh::mesh_t>(mesh);
}



std::shared_ptr<TerrainMesh::mesh_t> TerrainMesh::createMeshCenter(){
    int m = this->m*2;
    return createGridMesh(m,m,vec2(1.0/(m-2)),vec2(0.5));
}


std::shared_ptr<TerrainMesh::mesh_t> TerrainMesh::createGridMesh(unsigned int w, unsigned int h, vec2 d, vec2 o){
    mesh_t* mesh = new mesh_t();

    //creating uniform grid with w*h vertices
    //the resulting mesh will fill the quad (-1,0,-1) - (1,0,1)
    float dw = d.x;
    float dh = d.y;
    for(unsigned int y=0;y<h;y++){
        for(unsigned int x=0;x<w;x++){
            float fx = (float)x*dw-o.x;
            float fy = (float)y*dh-o.y;
            Vertex v(vec3(fx,0.0f,fy));
            mesh->addVertex(v);
        }
    }


    for(unsigned int y=0;y<h-1;y++){
        for(unsigned int x=0;x<w-1;x++){
            GLuint quad[] = {y*w+x,(y+1)*w+x,(y+1)*w+x+1,y*w+x+1};
            mesh->addQuad(quad);
        }
    }

    return std::shared_ptr<TerrainMesh::mesh_t>(mesh);
}
