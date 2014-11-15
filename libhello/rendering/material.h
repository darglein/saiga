#pragma once

#include <string>
#include <fstream>

#include "libhello/util/glm.h"
#include "libhello/opengl/texture.h"
#include "libhello/util/loader.h"

class Material{
public:
    std::string name;
    float Ns = 0.2;   //specular coeffizient
    float Ni = 1;
    float d = 1;    //transparency
    float Tr = 1;
    vec3 Tf;
    int illum ;
    vec3 Ka = vec3(0.2);   //ambient color
    vec3 Kd = vec3(0.8);   //diffuse color
    vec3 Ks = vec3(1);   //specular color
    vec3 Ke = vec3(1);
    Texture* map_Ka = NULL;
    Texture* map_Kd = NULL;
    Texture* map_Ks = NULL;
    Texture* map_d = NULL ;
    Texture* map_bump = NULL ;

    friend std::ostream& operator<< (std::ostream& stream, const Material& material);


};

struct TriangleGroup{
    int startFace;
    int faces;
    Material* mat;
};



class MaterialLoader : public Loader<Material>{
public:
    TextureLoader* textureLoader;
    Material* currentMaterial = 0;
    void parseLine(char* line);

    MaterialLoader(){}
    Material* load(const std::string &name);
    Material* loadFromFile(const std::string &name){(void)name;return nullptr;}
    void loadLibrary(const std::string &file);
    bool loadLibraryFromFile(const std::string &path);
};


