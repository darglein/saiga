/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once


#include "saiga/config.h"
#include "saiga/core/util/math.h"

#include <vector>

namespace Saiga
{
/**
 * Similar to the obj material.
 *
 * It contains lots of stuff. The loader decides which members are loaded.
 * The renderer decides how this material is rendered.
 *
 * TODO: Extend it with more stuff
 */
struct SAIGA_CORE_API GenericMaterial
{
    std::string name;
    vec4 color_diffuse  = vec4(0, 1, 0, 0);
    vec4 color_ambient  = vec4(0, 1, 0, 0);
    vec4 color_specular = vec4(0, 1, 0, 0);

    std::string texture_diffuse;
};

struct SAIGA_CORE_API GenericModel
{
   public:
    GenericModel() {}
    GenericModel(const std::string& file) { load(file); }

    bool load(const std::string& file);


   protected:
    bool isTriangleModel = false;

    // Vertex Data
    std::vector<vec3> position;
    std::vector<vec3> normal;
    std::vector<vec2> texCoords;

    // Face Data

    // Vertex indices
    std::vector<int> indices;
    struct GenericFace
    {
        // Pointers into the indices array above
        int startIndex, endIndex;
        int size() { return endIndex - startIndex; }
    };
    std::vector<GenericFace> faces;

    // Materials

    std::vector<GenericMaterial> materials;

    // Defines which faces have which materials
    struct MaterialGroup
    {
        int startFace = 0;
        int numFaces  = 0;
        int materialId;
    };
    std::vector<MaterialGroup> materialGroups;
};


// class SAIGA_CORE_API ModelLoader
//{
//   public:
//    void load(const std::string& file);

// private:
//    GenericModel model;
//};



}  // namespace Saiga
