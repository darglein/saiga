#pragma once

#include "libhello/opengl/texture/texture.h"

#include "libhello/util/loader.h"
#include "libhello/util/singleton.h"


class TextureLoader : public Loader<Texture>, public Singleton <TextureLoader>{
    friend class Singleton <TextureLoader>;
public:
    Texture* loadFromFile(const std::string &name);
};

