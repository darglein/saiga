/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/geometry/triangle_mesh.h"


namespace Saiga
{
struct SAIGA_CORE_API ObjMaterial
{
    std::string name;
    vec4 color = make_vec4(0, 1, 0, 0);

    float Ns = 0.2f;  // specular coefficient
    float Ni = 1;
    float d  = 1;  // transparency
    float Tr = 1;
    vec3 Tf;
    int illum;
    vec3 Ka = make_vec3(0.2f);  // ambient color
    vec4 Kd = make_vec4(0.8f);  // diffuse color
    vec3 Ks = make_vec3(1);     // specular color
    vec3 Ke = make_vec3(1);

    std::string map_Ka;
    std::string map_Kd;
    std::string map_Ks;
    std::string map_d;
    std::string map_bump;

    ObjMaterial(const std::string& name = "") : name(name) {}
};


class SAIGA_CORE_API ObjMaterialLoader
{
   public:
    std::string file;
    bool verbose = false;

   public:
    ObjMaterialLoader() {}
    ObjMaterialLoader(const std::string& file);



    bool loadFile(const std::string& file);

    ObjMaterial getMaterial(const std::string& name);

   private:
    std::vector<ObjMaterial> materials;
    ObjMaterial* currentMaterial = nullptr;
    void parseLine(const std::string& line);

    // tries to find a file from the material (for example a texture)
    // returns the absolute path to that file
    std::string getPathImage(const std::string& s);
};

}  // namespace Saiga
