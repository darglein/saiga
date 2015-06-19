#include "rendering/material.h"

std::ostream& operator<< (std::ostream& stream, const Material& material){
    stream<<material.name<<material.Kd<<material.Ka;
    return stream;
}

Material* MaterialLoader::load(const std::string &name){

    return exists(name);

//    //check if already exists
//    for(Material* &object : objects){
//        if(object->name == name)
//            return object;
//    }
////    cout<<"Could not load Material: "<<name<<endl;
//    return NULL;
}

void MaterialLoader::loadLibrary(const std::string &file){
    bool erg;
    for(std::string &path : locations){
        std::string complete_path = path + "/" + file;
        erg = loadLibraryFromFile(complete_path);
        if (erg){
            return;
        }
    }
    cout<<"Warning Material Library "<<file<<" not loaded!"<<endl;
    cout<<"Make sure you have set the correct Material Library path."<<endl;
}

bool MaterialLoader::loadLibraryFromFile(const std::string &path){
    std::fstream stream;
    stream.exceptions ( std::fstream::failbit | std::fstream::badbit );


    try {
        stream.open (path, std::fstream::in);
    }
    catch (std::fstream::failure e) {
        return false;
    }
    currentMaterial = 0;

    cout<<"mtlloader: loading file "<<path<<endl;

    char buffer[1024];
    string line;
    string header;
    try {
        while(stream.getline(buffer,1024)){ //lines
            parseLine(buffer);
        }

    }
    catch (std::fstream::failure &e) {
//        cout<<"MTL Loading finished"<<endl;
    }
    return true;

}

extern bool skipTokens(char* &str);
extern bool readUntilToken(char* &str,string &out_value);
extern vec3 readVec3(char* &str);

void MaterialLoader::parseLine(char* line){
    string header;
    skipTokens(line);
    readUntilToken(line,header);
    skipTokens(line);

    if(header == "newmtl"){
//        cout<<"new material: "<<line<<endl;
        currentMaterial = new Material();
//        objects.push_back(currentMaterial);
        objects.emplace_back(line,NoParams(),currentMaterial);
        currentMaterial->name = line;
    }
    if(currentMaterial==0){
//        cout<<"mtl error: no material created"<<endl;
        return;
    }

    if(header == "Ns"){
        currentMaterial->Ns = atof(line);
    }else if(header == "Ni"){
        currentMaterial->Ni = atof(line);
    }else if(header == "d"){
        currentMaterial->d = atof(line);
    }else if(header == "Tr"){
        currentMaterial->Tr = atof(line);
    }else if(header == "Tf"){
        currentMaterial->Tf = readVec3(line);
    }else if(header == "illum"){
        currentMaterial->illum = atoi(line);
    }else if(header == "Ka"){
        currentMaterial->Ka = readVec3(line);
    }else if(header == "Kd"){
        currentMaterial->Kd = readVec3(line);
    }else if(header == "Ks"){
        currentMaterial->Ks = readVec3(line);
    }else if(header == "Ke"){
        currentMaterial->Ke = readVec3(line);
    }else if(header == "map_Ka"){
        currentMaterial->map_Ka = textureLoader->load(line);
        currentMaterial->map_Ka->setWrap(GL_REPEAT);
    }else if(header == "map_Kd"){
        currentMaterial->map_Kd = textureLoader->load(line);
        currentMaterial->map_Kd->setWrap(GL_REPEAT);
    }else if(header == "map_Ks"){
        currentMaterial->map_Ks = textureLoader->load(line);
         currentMaterial->map_Ks->setWrap(GL_REPEAT);
    }else if(header == "map_d"){
        currentMaterial->map_d = textureLoader->load(line);
         currentMaterial->map_d->setWrap(GL_REPEAT);
    }else if(header == "map_bump" || header == "bump"){
        currentMaterial->map_bump = textureLoader->load(line);
         currentMaterial->map_bump->setWrap(GL_REPEAT);
    }
}


