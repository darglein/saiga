#pragma once

#include <saiga/assets/assetLoader.h>
#include <saiga/assets/coloredAsset.h>

class SAIGA_GLOBAL AssimpAssetLoader : public AssetLoader2{
public:


    AssimpAssetLoader();
    virtual ~AssimpAssetLoader();

    ColoredAsset* loadBasicAsset(const std::string &file, bool normalize=false);
    TexturedAsset* loadTexturedAsset(const std::string &file, bool normalize=false);
    AnimatedAsset* loadAnimatedAsset(const std::string &file);


    Asset* loadAsset(const std::string &file);
};
