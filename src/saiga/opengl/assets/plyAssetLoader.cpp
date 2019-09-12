/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "plyAssetLoader.h"

#include "saiga/core/model/plyModelLoader.h"
#include "saiga/opengl/shader/shaderLoader.h"
#include "saiga/opengl/texture/TextureLoader.h"

namespace Saiga
{
void PlyAssetLoader::loadMeshNC(const std::string& file, TriangleMesh<VertexNC, GLuint>& mesh, bool normalize)
{
    PLYLoader loader(file);
    mesh = loader.mesh;
}

std::shared_ptr<ColoredAsset> PlyAssetLoader::loadBasicAsset(const std::string& file, bool normalize)
{
    TriangleMesh<VertexNC, GLuint> mesh;
    loadMeshNC(file, mesh, normalize);

    return assetFromMesh(mesh);
}



}  // namespace Saiga
