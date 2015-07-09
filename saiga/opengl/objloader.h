#pragma once

#include <vector>
#include <string>

#include "saiga/opengl/vertex.h"
#include "saiga/opengl/vertexBuffer.h"
#include "saiga/opengl/mesh.h"
#include "saiga/util/loader.h"
#include "saiga/rendering/material.h"
#include "saiga/geometry/aabb.h"
#include "saiga/geometry/triangle_mesh.h"
#include "saiga/geometry/material_mesh.h"

struct SAIGA_GLOBAL IndexedVertex{
    int v,n,t;
};

struct SAIGA_GLOBAL Face{
    IndexedVertex vertices[3];
    Face(){}

};




typedef MaterialMesh<VertexNT,GLuint> material_mesh_t;

class SAIGA_GLOBAL ObjLoader : public Loader<material_mesh_t>, public Singleton <ObjLoader>{
 friend class Singleton <ObjLoader>;
public:
//    MaterialLoader* materialLoader;

    int state;
    int maxCorners;

    //temporary vectors
    std::vector<vec3> vertices;
    std::vector<vec3> normals;
    std::vector<vec2> texCoords;
    std::vector<Face> faces;
    //        std::vector<unsigned int> indices;
    std::vector<bool> vertices_used;



//    std::vector<unsigned int> outIndices;
//    std::vector<VertexNT> outVertices;

    std::vector<TriangleGroup> triangleGroups;

    ObjLoader():state(0),maxCorners(0){}
    virtual ~ObjLoader(){}
    void reset();
    void parseLine(char* line);
    material_mesh_t *createOutput();

    void parseV(char* line);
    void parseN(char* line);
    void parseT(char* line);
    void parseF(char* line);

    bool extractIndices(char* line, int &v1, int &v2, int &v3);

    material_mesh_t* loadFromFile(const std::string &name, const NoParams &params);
};

