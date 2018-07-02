/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "AssetRenderer.h"
#include "saiga/vulkan/Shader/all.h"
#include "saiga/vulkan/Vertex.h"
#include "saiga/animation/objLoader2.h"



namespace Saiga {
namespace Vulkan {

void VertexColoredAsset::load(const std::string &file)
{
    Saiga::ObjLoader2 loader(file);
    loader.computeVertexColorAndData();
    loader.toTriangleMesh(mesh);
}

}
}
