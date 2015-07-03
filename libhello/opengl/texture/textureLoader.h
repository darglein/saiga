#pragma once

#include "libhello/opengl/texture/texture.h"

#include "libhello/util/loader.h"
#include "libhello/util/singleton.h"


struct SAIGA_GLOBAL TextureParameters{
    bool srgb = true;
};

SAIGA_GLOBAL bool operator==(const TextureParameters& lhs, const TextureParameters& rhs);


class SAIGA_GLOBAL TextureLoader : public Loader<Texture,TextureParameters>, public Singleton <TextureLoader>{
    friend class Singleton <TextureLoader>;
public:
    Texture* loadFromFile(const std::string &name, const TextureParameters &params);
};

