#pragma once

#include "saiga/util/color.h"
#include <saiga/assets/coloredAsset.h>
#include <saiga/assets/animatedAsset.h>
#include "saiga/animation/boneShader.h"

class SAIGA_GLOBAL AssetLoader2{
public:
    MVPShader* basicAssetShader  = nullptr;
    MVPShader* basicAssetDepthshader  = nullptr;
    MVPShader* basicAssetWireframeShader  = nullptr;

    MVPShader* texturedAssetShader  = nullptr;
    MVPShader* texturedAssetDepthShader  = nullptr;
    MVPShader* texturedAssetWireframeShader  = nullptr;

    BoneShader* animatedAssetShader  = nullptr;
    BoneShader* animatedAssetDepthshader  = nullptr;
    BoneShader* animatedAssetWireframeShader  = nullptr;

    AssetLoader2();
    virtual ~AssetLoader2();

    void loadDefaultShaders();

    /**
     * Creates a plane with a checker board texture.
     * The plane lays in the x-z plane with a normal pointing to positve y.
     * size.x and size.y are the dimensions of the plane.
     * quadSize is the size of one individual quad of the checkerboard.
     */

    TexturedAsset* loadDebugPlaneAsset(vec2 size, float quadSize=1.0f, Color color1=Colors::lightgray, Color color2=Colors::gray);
};
