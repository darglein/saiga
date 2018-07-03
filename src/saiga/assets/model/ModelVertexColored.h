/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/assets/model/Model.h"

namespace Saiga {


class SAIGA_GLOBAL VertexColoredModel : public Model<VertexNC,uint32_t>
{
public:

    void createCheckerBoard(glm::ivec2 size, float quadSize, vec4 color1, vec4 color2);
    void loadObj(const std::string &file);
};


class SAIGA_GLOBAL TexturedModel : public Model<VertexNTD,uint32_t>
{
public:


};




}
