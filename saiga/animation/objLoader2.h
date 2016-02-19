#pragma once

#include <saiga/config.h>
#include <saiga/geometry/triangle_mesh.h>
#include <saiga/opengl/texture/texture.h>

struct SAIGA_GLOBAL IndexedVertex2{
    int v=-1,n=-1,t=-1;
};

struct SAIGA_GLOBAL ObjMaterial{
    vec3 color = vec3(0,1,0);
    Texture* diffuseTexture = nullptr;
};

struct SAIGA_GLOBAL ObjTriangleGroup{
    int startFace = 0;
    int faces = 0;
    ObjMaterial material;
};

struct SAIGA_GLOBAL ObjTriangle{
    GLuint v[3];
};

class SAIGA_GLOBAL ObjLoader2{
public:
    std::string file;
    bool verbose = false;

public:
    ObjLoader2(){}
    ObjLoader2(const std::string &file);



    bool loadFile(const std::string &file);


    std::vector<VertexNT> outVertices;
    std::vector<ObjTriangle> outTriangles;
    std::vector<ObjTriangleGroup> triangleGroups;
    void separateVerticesByGroup();
private:
    std::vector<vec3> vertices;
    std::vector<vec3> normals;
    std::vector<vec2> texCoords;
    std::vector<std::vector<IndexedVertex2>> faces;


    void createVertexIndexList();

    std::vector<std::vector<IndexedVertex2>> triangulateFace(const std::vector<IndexedVertex2> &face);

    void parseLine(const std::string &line);

    void parseV(const std::string &line);
    void parseVT(const std::string &line);
    void parseVN(const std::string &line);
    void parseF(std::string &line);
    IndexedVertex2 parseIV(std::string &line);

    void parseUM(const std::string &line);
    void parseM(const std::string &line);

};


