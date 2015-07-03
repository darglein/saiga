#pragma once

#include "libhello/opengl/texture/raw_texture.h"


/*
 *  From Stackoverflow:
 *
 * Cube Maps have been specified to follow the RenderMan specification (for whatever reason),
 *  and RenderMan assumes the images' origin being in the upper left, contrary to the usual
 * OpenGL behaviour of having the image origin in the lower left. That's why things get swapped
 *  in the Y direction. It totally breaks with the usual OpenGL semantics and doesn't make
 * sense at all. But now we're stuck with it.
 *
 * -> Swap Y before creating a cube texture from a image
 */
class SAIGA_GLOBAL cube_Texture : public raw_Texture{

public:
    cube_Texture():raw_Texture(GL_TEXTURE_CUBE_MAP){}
    virtual ~cube_Texture(){}


     void setDefaultParameters() override;


    void uploadData(GLenum target, GLubyte* data );
    void uploadData(GLubyte** data );
    void uploadData(GLubyte* data);

    bool fromImage(Image *img);
    bool fromImage(Image &img);
};

