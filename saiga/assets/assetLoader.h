#pragma once

#include <saiga/assets/coloredAsset.h>

class SAIGA_GLOBAL AssetLoader2{
public:
    MVPShader* basicAssetShader  = nullptr;
    MVPShader* basicAssetDepthshader  = nullptr;

    MVPShader* texturedAssetShader  = nullptr;
    MVPShader* texturedAssetDepthShader  = nullptr;

    AssetLoader2();
    virtual ~AssetLoader2();

    void loadDefaultShaders();
};
