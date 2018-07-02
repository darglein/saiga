/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#pragma once

#include "saiga/geometry/triangle_mesh.h"


namespace Saiga {
namespace Vulkan {

class SAIGA_GLOBAL VertexColoredAsset
{
protected:
    Saiga::TriangleMesh<Saiga::VertexNC, uint32_t> mesh;


public:
    void load(const std::string &file);

};


}
}
