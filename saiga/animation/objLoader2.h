#pragma once

#include <saiga/config.h>
#include <saiga/geometry/triangle_mesh.h>


struct SAIGA_GLOBAL IndexedVertex2{
    int v=-1,n=-1,t=-1;
};

class SAIGA_GLOBAL ObjLoader2{
public:
    std::string file;
    bool verbose = false;

public:
    ObjLoader2(){}
    ObjLoader2(const std::string &file);



    bool loadFile(const std::string &file);
private:
    std::vector<vec3> vertices;
    std::vector<vec3> normals;
    std::vector<vec2> texCoords;


    void parseLine(const std::string &line);

    void parseV(const std::string &line);
    void parseVT(const std::string &line);
    void parseVN(const std::string &line);
    void parseF(std::string &line);
    IndexedVertex2 parseIV(std::string &line);

    void parseUM(const std::string &line);
    void parseM(const std::string &line);

};


