#include "saiga/animation/objLoader2.h"

#include <fstream>
#include <sstream>
#include <algorithm>


std::vector<std::string> &split(const std::string &s, char delim, std::vector<std::string> &elems);

std::vector<std::string> split(const std::string &s, char delim);


//======================================================================

ObjLoader2::ObjLoader2(const std::string &file):file(file)
{
    loadFile(file);
}

bool ObjLoader2::loadFile(const std::string &file){

    std::ifstream stream(file, std::ios::in);
    if(!stream.is_open()) {
        return false;
    }


    cout<<"objloader: loading file "<<file<<endl;


    while(!stream.eof()) {
        std::string line;
        std::getline(stream, line);
        parseLine(line);
    }

    cout<<"objloader finished :)"<<endl;
    return true;
}

void ObjLoader2::parseLine(const std::string &line)
{
    std::stringstream sstream(line);

    std::string header;
    sstream >> header;

    std::string rest;
    std::getline(sstream,rest);


    if(header == "#"){
    }else if(header == "usemtl"){
        parseUM(rest);
    }else if(header == "mtllib"){
        parseM(rest);
    }else if(header == "g"){
        //        cout<<"Found Group: "<<line<<endl;
    }else if(header == "o"){
        //        cout<<"Found Object: "<<line<<endl;
    }else if(header == "s"){
        //smooth shading
    }else if(header == "v"){
        parseV(rest);
    }else if(header == "vt"){
        parseVT(rest);
    }else if(header == "vn"){
        parseVN(rest);
    }else if(header == "f"){
        parseF(rest);
    }
}

void ObjLoader2::parseV(const std::string &line)
{
    std::stringstream sstream(line);
    vec3 v;
    sstream >> v.x >> v.y >> v.z;
    vertices.push_back(v);
}

void ObjLoader2::parseVT(const std::string &line)
{
    std::stringstream sstream(line);
    vec2 v;
    sstream >> v.x >> v.y;
    texCoords.push_back(v);
}

void ObjLoader2::parseVN(const std::string &line)
{
    std::stringstream sstream(line);
    vec3 v;
    sstream >> v.x >> v.y >> v.z;
    normals.push_back(glm::normalize(v));
}

void ObjLoader2::parseF(std::string &line)
{

//    std::replace( line.begin(), line.end(), '/', ' ');

    std::stringstream sstream(line);
    int t1, t2 , t3;
    std::string t;

    cout<<"parse F "<<line<<endl;

    while(sstream >> t){
        IndexedVertex2 iv = parseIV(t);
    }

    cout<<endl<<endl;
}

//parsing index vertex
//examples:
//v1/vt1/vn1        12/51/1
//v1//vn1           51//4
IndexedVertex2 ObjLoader2::parseIV(std::string &line)
{
    IndexedVertex2 iv;
    std::vector<std::string> s = split(line, '/');
    if(s.size()>0)
        iv.v = std::atoi(s[0].c_str());
    if(s.size()>1)
        iv.t = std::atoi(s[1].c_str());
    if(s.size()>2)
        iv.n = std::atoi(s[2].c_str());
    return iv;
}

void ObjLoader2::parseUM(const std::string &line)
{

}

void ObjLoader2::parseM(const std::string &line)
{

}


