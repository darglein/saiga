#include "libhello/geometry/triangle_mesh_generator.h"



std::shared_ptr<TriangleMesh<VertexNT,GLuint>> TriangleMeshGenerator::createMesh(const Sphere &sphere, int rings, int sectors){

    TriangleMesh<VertexNT,GLuint>* mesh = new TriangleMesh<VertexNT,GLuint>();
    float const R = 1./(float)(rings);
    float const S = 1./(float)(sectors);
    int r, s;

    for(r = 0; r < rings+1; r++){
        for(s = 0; s < sectors; s++) {
            float y = glm::sin( -M_PI_2 + M_PI * r * R );
            float x = glm::cos(2*M_PI * s * S) * glm::sin( M_PI * r * R );
            float z = glm::sin(2*M_PI * s * S) * glm::sin( M_PI * r * R );

            VertexNT vert;
            vert.texture = vec2(s*S,r*R);
            vert.position = vec3(x, y, z );
            vert.normal = vec3( x,y, z);
            mesh->vertices.push_back( vert);
        }
    }


    for(r = 0; r < rings; r++){
        for(s = 0; s < sectors; s++) {
            if(r!=rings-1){
                Face face;
                face.v1 =  (r+1) * sectors + s;
                face.v2 =  (r+1) * sectors + (s+1)%sectors;
                face.v3 =   r * sectors + (s+1)%sectors;
                mesh->faces.push_back(face);
            }
            if(r!=0){
                Face face;
                face.v1 =  (r+1) * sectors + s;
                face.v2 =   r * sectors + (s+1)%sectors;
                face.v3 =  r * sectors + s;
                mesh->faces.push_back(face);
            }
        }
    }

    return std::shared_ptr<TriangleMesh<VertexNT,GLuint>>(mesh);
}

std::shared_ptr<TriangleMesh<VertexNT,GLuint>> TriangleMeshGenerator::createMesh(const Sphere &sphere, int resolution){
    TriangleMesh<VertexNT,GLuint>* mesh = new TriangleMesh<VertexNT,GLuint>();
    float t = (1.0 + glm::sqrt(5.0)) / 2.0;


    mesh->vertices.push_back(VertexNT(vec3(-1,  t,  0),vec3(),vec2()));
    mesh->vertices.push_back(VertexNT(vec3( 1,  t,  0),vec3(),vec2()));
    mesh->vertices.push_back(VertexNT(vec3(-1, -t,  0),vec3(),vec2()));
    mesh->vertices.push_back(VertexNT(vec3( 1, -t,  0),vec3(),vec2()));

    mesh->vertices.push_back(VertexNT(vec3( 0, -1,  t),vec3(),vec2()));
    mesh->vertices.push_back(VertexNT(vec3( 0,  1,  t),vec3(),vec2()));
    mesh->vertices.push_back(VertexNT(vec3( 0, -1, -t),vec3(),vec2()));
    mesh->vertices.push_back(VertexNT(vec3( 0,  1, -t),vec3(),vec2()));

    mesh->vertices.push_back(VertexNT(vec3( t,  0, -1),vec3(),vec2()));
    mesh->vertices.push_back(VertexNT(vec3( t,  0,  1),vec3(),vec2()));
    mesh->vertices.push_back(VertexNT(vec3(-t,  0, -1),vec3(),vec2()));
    mesh->vertices.push_back(VertexNT(vec3(-t,  0,  1),vec3(),vec2()));

    for(VertexNT &v : mesh->vertices){
        v.position = glm::normalize(v.position);
        v.normal = v.position;
    }

    mesh->faces.push_back(Face(0,11,5));
    mesh->faces.push_back(Face(0,5,1));
    mesh->faces.push_back(Face(0,1,7));
    mesh->faces.push_back(Face(0,7,10));
    mesh->faces.push_back(Face(0,10,11));

    mesh->faces.push_back(Face(1,5,9));
    mesh->faces.push_back(Face(5,11,4));
    mesh->faces.push_back(Face(11, 10, 2));
    mesh->faces.push_back(Face(10, 7, 6));
    mesh->faces.push_back(Face(7, 1, 8));

    mesh->faces.push_back(Face(3,9,4));
    mesh->faces.push_back(Face(3,4,2));
    mesh->faces.push_back(Face(3, 2, 6));
    mesh->faces.push_back(Face(3, 6, 8));
    mesh->faces.push_back(Face(3, 8, 9));

    mesh->faces.push_back(Face(4,9,5));
    mesh->faces.push_back(Face(2,4,11));
    mesh->faces.push_back(Face(6, 2, 10));
    mesh->faces.push_back(Face(8, 6, 7));
    mesh->faces.push_back(Face(9, 8, 1));


    for(int r=0;r<resolution;r++){
        int faces = mesh->faces.size();
        for(int i=0;i<faces;i++){
            mesh->subdivideFace(i);
        }

        for(VertexNT &v : mesh->vertices){
            v.position = glm::normalize(v.position);
            v.normal = v.position;
        }
    }


    return std::shared_ptr<TriangleMesh<VertexNT,GLuint>>(mesh);
}

std::shared_ptr<TriangleMesh<VertexNT,GLuint>> TriangleMeshGenerator::createQuadMesh(){

    TriangleMesh<VertexNT,GLuint>* mesh = new TriangleMesh<VertexNT,GLuint>();
    mesh->vertices.push_back(VertexNT(vec3(0,0,0),vec3(0,1,0),vec2(0,0)));
    mesh->vertices.push_back(VertexNT(vec3( 1,0,0),vec3(0,1,0),vec2(1,0)));
    mesh->vertices.push_back(VertexNT(vec3(1,0,1),vec3(0,1,0),vec2(1,1)));
    mesh->vertices.push_back(VertexNT(vec3(0,0,1),vec3(0,1,0),vec2(0,1)));

    mesh->faces.push_back(Face(0,2,1));
    mesh->faces.push_back(Face(0,3,2));
    return std::shared_ptr<TriangleMesh<VertexNT,GLuint>>(mesh);
}

std::shared_ptr<TriangleMesh<VertexNT,GLuint>> TriangleMeshGenerator::createFullScreenQuadMesh(){
    TriangleMesh<VertexNT,GLuint>* mesh = new TriangleMesh<VertexNT,GLuint>();
    mesh->vertices.push_back(VertexNT(vec3(-1,-1,0),vec3(0,0,1),vec2(0,0)));
    mesh->vertices.push_back(VertexNT(vec3( 1,-1,0),vec3(0,0,1),vec2(1,0)));
    mesh->vertices.push_back(VertexNT(vec3(1,1,0),vec3(0,0,1),vec2(1,1)));
    mesh->vertices.push_back(VertexNT(vec3(-1,1,0),vec3(0,0,1),vec2(0,1)));

    mesh->faces.push_back(Face(0,2,3));
    mesh->faces.push_back(Face(0,1,2));
    return std::shared_ptr<TriangleMesh<VertexNT,GLuint>>(mesh);
}

std::shared_ptr<TriangleMesh<VertexNT,GLuint>> TriangleMeshGenerator::createMesh(const Plane &plane){
    TriangleMesh<VertexNT,GLuint>* mesh = new TriangleMesh<VertexNT,GLuint>();
    mesh->vertices.push_back(VertexNT(vec3(-1,0,-1),vec3(0,1,0),vec2(0,0)));
    mesh->vertices.push_back(VertexNT(vec3( 1,0,-1),vec3(0,1,0),vec2(1,0)));
    mesh->vertices.push_back(VertexNT(vec3(1,0,1),vec3(0,1,0),vec2(1,1)));
    mesh->vertices.push_back(VertexNT(vec3(-1,0,1),vec3(0,1,0),vec2(0,1)));

    mesh->faces.push_back(Face(0,2,1));
    mesh->faces.push_back(Face(0,3,2));
    return std::shared_ptr<TriangleMesh<VertexNT,GLuint>>(mesh);
}

std::shared_ptr<TriangleMesh<VertexNT, GLuint>> TriangleMeshGenerator::createMesh(const Cone &cone, int sectors){
    TriangleMesh<VertexNT, GLuint>* mesh = new TriangleMesh<VertexNT, GLuint>();
    mesh->vertices.push_back(VertexNT(vec3(0,0,0),vec3(0,1,0),vec2(0,0)));  //top
    mesh->vertices.push_back(VertexNT(vec3(0,-cone.height,0),vec3(0,-1,0),vec2(0,0)));  //bottom

    float const R = 1./(float)(sectors);
    float const r = 1;//glm::tan(glm::radians( cone.alpha))*cone.height; //radius

    for(int s=0;s<sectors;s++){
        float x = r * glm::sin((float)s*R*M_PI*2.0f);
        float y = r * glm::cos((float)s*R*M_PI*2.0f);
        mesh->vertices.push_back(VertexNT(vec3(x,-cone.height,y),glm::normalize(vec3(x,0,y)),vec2(0,0)));
    }

    for(int s=0;s<sectors;s++){
        Face face;
        face.v1 =  s+2;
        face.v2 =  ((s+1)%sectors)+2;
        face.v3 =  0;
        mesh->faces.push_back(face);

        face.v1 =  1;
        face.v2 =  ((s+1)%sectors)+2;
        face.v3 =  s+2;
        mesh->faces.push_back(face);
    }

    return std::shared_ptr<TriangleMesh<VertexNT, GLuint>>(mesh);
}


