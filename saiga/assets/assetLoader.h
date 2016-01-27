#pragma once

#include "saiga/util/color.h"
#include <saiga/assets/coloredAsset.h>

class SAIGA_GLOBAL AssetLoader2{
public:
    MVPShader* basicAssetShader  = nullptr;
    MVPShader* basicAssetDepthshader  = nullptr;
    MVPShader* basicAssetWireframeShader  = nullptr;

    MVPShader* texturedAssetShader  = nullptr;
    MVPShader* texturedAssetDepthShader  = nullptr;
    MVPShader* texturedAssetWireframeShader  = nullptr;

    AssetLoader2();
    virtual ~AssetLoader2();

    void loadDefaultShaders();

    TexturedAsset* loadDebugPlaneAsset(vec2 size, Color color1, Color color2);
};
