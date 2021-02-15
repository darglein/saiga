/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/assets/assetLoader.h"

namespace Saiga
{
class SAIGA_OPENGL_API ObjAssetLoader : public AssetLoader
{
   public:
    ObjAssetLoader();
    virtual ~ObjAssetLoader();

    //    void loadMeshNC(const std::string& file, TriangleMesh<VertexNC, GLuint>& mesh, bool normalize = false);

    std::shared_ptr<ColoredAsset> loadColoredAsset(const std::string& file, bool normalize = false);
    std::shared_ptr<TexturedAsset> loadTexturedAsset(const std::string& file, bool normalize = false);
};

}  // namespace Saiga
