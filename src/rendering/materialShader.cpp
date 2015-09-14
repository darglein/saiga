#include "saiga/rendering/materialShader.h"


void MaterialShader::checkUniforms(){
    MVPShader::checkUniforms();
    location_colors = getUniformLocation("colors");
    location_textures = getUniformLocation("textures");
    location_use_textures = getUniformLocation("use_textures");
    textures[0] = 0;
    textures[1] = 1;
    textures[2] = 2;
    textures[3] = 3;
    textures[4] = 4;
}

void MaterialShader::uploadMaterial(const Material &material){
    colors[0] = material.Ka;
    colors[1] = material.Kd;
    colors[2] = material.Ks;
    upload(location_colors,3,colors);
    for(int i=0;i<5;i++)
        use_textures[i] = 0.0f;

    if(material.map_Ka){
        material.map_Ka->bind(0);
        use_textures[0] = 1.0f;
    }
    if(material.map_Kd){
        material.map_Kd->bind(1);
        use_textures[1] = 1.0f;
    }
    if(material.map_Ks){
        material.map_Ks->bind(2);
        use_textures[2] = 1.0f;
    }
    if(material.map_d){
        material.map_d->bind(3);
        use_textures[3] = 1.0f;
    }
    if(material.map_bump){
        material.map_bump->bind(4);
        use_textures[4] = 1.0f;
    }
    glUniform1iv(location_textures,5, textures);
    glUniform1fv(location_use_textures,5, use_textures);
}
