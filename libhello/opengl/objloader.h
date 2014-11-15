#ifndef OBJLOADER_H
#define OBJLOADER_H

#include <vector>
#include <fstream>
#include <iostream>
#include <string.h>
#include "libhello/opengl/vertex.h"
#include "libhello/opengl/vertexBuffer.h"
#include "libhello/opengl/mesh.h"
#include "libhello/util/loader.h"
#include "libhello/rendering/material.h"
#include "libhello/geometry/aabb.h"


struct IndexedVertex{
    int v,n,t;
};

struct Face{
    IndexedVertex vertices[3];
    Face(){}

};





class ObjLoader : public Loader<MaterialMesh>{
public:
    MaterialLoader* materialLoader;

    int state;
    int maxCorners;

    //temporary vectors
    std::vector<vec3> vertices;
    std::vector<vec3> normals;
    std::vector<vec2> texCoords;
    std::vector<Face> faces;
    //        std::vector<unsigned int> indices;
    std::vector<bool> vertices_used;

    std::vector<unsigned int> outIndices;
    std::vector<VertexNT> outVertices;
    std::vector<TriangleGroup> triangleGroups;

    ObjLoader():state(0),maxCorners(0){}
    void parseLine(char* line);
    void createOutput();

    void parseV(char* line);
    void parseN(char* line);
    void parseT(char* line);
    void parseF(char* line);

    bool extractIndices(char* line, int &v1, int &v2, int &v3);

    MaterialMesh* loadFromFile(const string &name);
};

#endif // OBJLOADER_H
