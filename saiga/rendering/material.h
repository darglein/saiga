#pragma once

#include <string>
#include <fstream>

#include "saiga/util/glm.h"
#include "saiga/opengl/texture/texture.h"
#include "saiga/util/loader.h"
#include "saiga/util/singleton.h"

class SAIGA_GLOBAL Material{
public:
    std::string name;
    float Ns = 0.2f;   //specular coefficient
    float Ni = 1;
    float d = 1;    //transparency
    float Tr = 1;
    vec3 Tf;
    int illum ;
    vec3 Ka = vec3(0.2f);   //ambient color
    vec3 Kd = vec3(0.8f);   //diffuse color
    vec3 Ks = vec3(1);   //specular color
    vec3 Ke = vec3(1);
    Texture* map_Ka = NULL;
    Texture* map_Kd = NULL;
    Texture* map_Ks = NULL;
    Texture* map_d = NULL ;
    Texture* map_bump = NULL ;

    friend std::ostream& operator<< (std::ostream& stream, const Material& material);


};

struct SAIGA_GLOBAL TriangleGroup{
    int startFace;
    int faces;
    Material* mat;
};



class SAIGA_GLOBAL MaterialLoader : public Loader<Material>, public Singleton <MaterialLoader>{
    friend class Singleton <MaterialLoader>;
public:
//    TextureLoader* textureLoader;
    Material* currentMaterial = 0;
    void parseLine(char* line);

    MaterialLoader(){}
    virtual ~MaterialLoader(){}
    Material* load(const std::string &name);
	Material* loadFromFile(const std::string &name, const NoParams &params){ (void)name; (void)params; return nullptr; }
    void loadLibrary(const std::string &file);
    bool loadLibraryFromFile(const std::string &path);
};


