#pragma once

#include <saiga/assets/coloredAsset.h>

class SAIGA_GLOBAL AssetLoader2{
public:
    MVPShader* shader  = nullptr;
    MVPShader* depthshader  = nullptr;

    MVPShader* textureshader  = nullptr;
    MVPShader* texturedepthshader  = nullptr;


    AssetLoader2();
    void loadDefaultShaders();

    ColoredAsset* loadBasicAsset(const std::string &file, bool normalize=false);
    TexturedAsset* loadTexturedAsset(const std::string &file, bool normalize=false);
    AnimatedAsset* loadAnimatedAsset(const std::string &file);

    Asset* test;

    Asset* loadAsset(const std::string &file);
};
