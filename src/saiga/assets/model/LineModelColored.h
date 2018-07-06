/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/assets/model/Model.h"

namespace Saiga {


class SAIGA_GLOBAL LineModelColored : public LineModel<VertexNC,uint32_t>
{
public:
    void createGrid(int numX, int numY, float quadSize=1.0f, vec4 color = vec4(0.7));
};



}
