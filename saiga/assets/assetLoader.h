#pragma once

#include <saiga/assets/coloredAsset.h>

class SAIGA_GLOBAL AssetLoader2{
public:
    AssetLoader2();

    ColoredAsset* loadBasicAsset(const std::string &file, bool normalize=false);
    AnimatedAsset* loadAnimatedAsset(const std::string &file);

    Asset* test;

    Asset* loadAsset(const std::string &file);
};
