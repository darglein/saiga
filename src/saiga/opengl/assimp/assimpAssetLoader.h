/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/assets/assetLoader.h"

namespace Saiga
{
class SAIGA_GLOBAL AssimpAssetLoader : public AssetLoader
{
   public:
    AssimpAssetLoader();
    virtual ~AssimpAssetLoader();

    std::shared_ptr<ColoredAsset> loadBasicAsset(const std::string& file, bool normalize = false);
    std::shared_ptr<TexturedAsset> loadTexturedAsset(const std::string& file, bool normalize = false);
    std::shared_ptr<AnimatedAsset> loadAnimatedAsset(const std::string& file, bool normalize = false);


    std::shared_ptr<Asset> loadAsset(const std::string& file);
};

}  // namespace Saiga
