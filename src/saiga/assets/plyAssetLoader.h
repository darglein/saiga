/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include <saiga/assets/assetLoader.h>

namespace Saiga
{
class SAIGA_GLOBAL PlyAssetLoader : public AssetLoader
{
   public:
    void loadMeshNC(const std::string& file, TriangleMesh<VertexNC, GLuint>& mesh, bool normalize = false);

    std::shared_ptr<ColoredAsset> loadBasicAsset(const std::string& file, bool normalize = false);
};

}  // namespace Saiga
