#pragma once

#include "libhello/opengl/raw_texture.h"

class cube_Texture : public raw_Texture{

public:
    cube_Texture():raw_Texture(GL_TEXTURE_CUBE_MAP){}
    virtual ~cube_Texture(){}


     void setDefaultParameters() override;


    //============= Required state: VALID =============

    void uploadData(GLubyte* data ) override;


};

