#pragma once

#include <saiga/assets/assetLoader.h>
#include <saiga/assets/coloredAsset.h>

class SAIGA_GLOBAL ObjAssetLoader : public AssetLoader2{
public:


    ObjAssetLoader();
    virtual ~ObjAssetLoader();

    ColoredAsset* loadBasicAsset(const std::string &file, bool normalize=false);

};
