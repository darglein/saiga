/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "Model.h"

namespace Saiga
{
class SAIGA_CORE_API VertexColoredModel : public TriangleModel<VertexNC, uint32_t>
{
   public:
    void createFullscreenQuad();

    void createCheckerBoard(ivec2 size, float quadSize, vec4 color1, vec4 color2);
    void loadObj(const std::string& file);
};


struct SAIGA_CORE_API Material
{
    std::string diffuse;
};


class SAIGA_CORE_API TexturedModel : public TriangleModel<VertexNTD, uint32_t>
{
   public:
    class SAIGA_CORE_API TextureGroup
    {
       public:
        int startIndex;
        int indices;
        Material material;
    };
    std::vector<TextureGroup> groups;


    void loadObj(const std::string& file);
};



}  // namespace Saiga
