/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <saiga/config.h>
#include "saiga/rendering/program.h"

namespace Saiga {


class SAIGA_GLOBAL Renderer{
public:
    Rendering* rendering = nullptr;


    Renderer();
    virtual ~Renderer();

    virtual void renderImGui(bool* p_open = NULL) = 0;
    virtual float getTotalRenderTime() = 0;

    void setRenderObject(Rendering &r );
};

}
