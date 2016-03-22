#pragma once

#include <saiga/assets/assetLoader.h>

class SAIGA_GLOBAL ObjAssetLoader : public AssetLoader2{
public:


    ObjAssetLoader();
    virtual ~ObjAssetLoader();

    ColoredAsset* loadBasicAsset(const std::string &file, bool normalize=false);
    TexturedAsset* loadTexturedAsset(const std::string &file, bool normalize=false);
};
