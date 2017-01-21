#pragma once

#include <saiga/assets/assetLoader.h>


class SAIGA_GLOBAL AssimpAssetLoader : public AssetLoader2{
public:


    AssimpAssetLoader();
    virtual ~AssimpAssetLoader();

    std::shared_ptr<ColoredAsset> loadBasicAsset(const std::string &file, bool normalize=false);
    std::shared_ptr<TexturedAsset> loadTexturedAsset(const std::string &file, bool normalize=false);
    std::shared_ptr<AnimatedAsset> loadAnimatedAsset(const std::string &file, bool normalize=false);


    std::shared_ptr<Asset> loadAsset(const std::string &file);
};
