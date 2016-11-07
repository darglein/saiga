#pragma once

#include "saiga/opengl/texture/texture3D.h"


/**
 * On CPU side the arraytexture behaves the same as a 3D texture.
 * The only difference is the Opengl target (GL_TEXTURE_2D_ARRAY instead of GL_TEXTURE_3D).
 */

class SAIGA_GLOBAL ArrayTexture2D : public Texture3D{
public:
    ArrayTexture2D():Texture3D(GL_TEXTURE_2D_ARRAY){}
    virtual ~ArrayTexture2D(){}
};

