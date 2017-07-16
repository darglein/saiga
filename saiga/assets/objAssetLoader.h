#pragma once

#include <saiga/assets/assetLoader.h>

namespace Saiga {

class SAIGA_GLOBAL ObjAssetLoader : public AssetLoader2{
public:


    ObjAssetLoader();
    virtual ~ObjAssetLoader();

    std::shared_ptr<ColoredAsset> loadBasicAsset(const std::string &file, bool normalize=false);
    std::shared_ptr<TexturedAsset> loadTexturedAsset(const std::string &file, bool normalize=false);
};

}
