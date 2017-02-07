#include "saiga/animation/objMaterialLoader.h"
#include "saiga/opengl/texture/textureLoader.h"
#include <fstream>
#include <sstream>
#include <algorithm>



ObjMaterialLoader::ObjMaterialLoader(const std::string &file):file(file)
{
    loadFile(file);
}

bool ObjMaterialLoader::loadFile(const std::string &_file){
    file = _file;
    std::ifstream stream(file, std::ios::in);
    if(!stream.is_open()) {
        return false;
    }


    cout<<"ObjMaterialLoader: loading file "<<file<<endl;


    while(!stream.eof()) {
        std::string line;
        std::getline(stream, line);
        //remove carriage return from windows
        line.erase( std::remove(line.begin(), line.end(), '\r'), line.end() );
        parseLine(line);
    }
    return true;
}

ObjMaterial ObjMaterialLoader::getMaterial(const std::string &name)
{
    for(ObjMaterial& m : materials){
        if(m.name == name)
            return m;
    }
    cout<<"Warning material '"<<name<<"' not found!"<<endl;
    return ObjMaterial("default");
}


void ObjMaterialLoader::parseLine(const std::string &line)
{
    std::stringstream sstream(line);

    std::string header;
    sstream >> header;

    std::string rest;
    std::getline(sstream,rest);

    //remove first white space
    if(rest[0]==' ' && rest.size()>1){
        rest = rest.substr(1);
    }

    std::stringstream restStream(rest);

    if(header == "newmtl"){
        ObjMaterial m(rest);
        materials.push_back(m);
        currentMaterial = &materials[materials.size()-1];

    }
    if(currentMaterial==nullptr){
        return;
    }

//    if(header == "Kd"){
//        restStream >> currentMaterial->color;
//    }

    if(header == "Ns"){
        restStream >> currentMaterial->Ns;
    }else if(header == "Ni"){
        restStream >> currentMaterial->Ni;
    }else if(header == "d"){
        restStream >> currentMaterial->d;
    }else if(header == "Tr"){
        restStream >> currentMaterial->Tr;
    }else if(header == "Tf"){
        restStream >> currentMaterial->Tf;
    }else if(header == "illum"){
        restStream >> currentMaterial->illum;
    }else if(header == "Ka"){
        restStream >> currentMaterial->Ka;
    }else if(header == "Kd"){
        restStream >> currentMaterial->Kd;
        currentMaterial->color = currentMaterial->Kd;
    }else if(header == "Ks"){
        restStream >> currentMaterial->Ks;
    }else if(header == "Ke"){
        restStream >> currentMaterial->Ke;
    }
    else if(header == "map_Ka"){
        currentMaterial->map_Ka = TextureLoader::instance()->load(rest);
        currentMaterial->map_Ka->setWrap(GL_REPEAT);
    }else if(header == "map_Kd"){
        currentMaterial->map_Kd = TextureLoader::instance()->load(rest);
        currentMaterial->map_Kd->setWrap(GL_REPEAT);
    }else if(header == "map_Ks"){
        currentMaterial->map_Ks = TextureLoader::instance()->load(rest);
         currentMaterial->map_Ks->setWrap(GL_REPEAT);
    }else if(header == "map_d"){
        currentMaterial->map_d = TextureLoader::instance()->load(rest);
         currentMaterial->map_d->setWrap(GL_REPEAT);
    }else if(header == "map_bump" || header == "bump"){
        currentMaterial->map_bump = TextureLoader::instance()->load(rest);
         currentMaterial->map_bump->setWrap(GL_REPEAT);
    }
}
