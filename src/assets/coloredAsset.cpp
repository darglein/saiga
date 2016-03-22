#include "saiga/assets/coloredAsset.h"


void TexturedAsset::render(Camera *cam, const mat4 &model)
{
    MVPTextureShader* shader = static_cast<MVPTextureShader*>(this->shader);
    shader->bind();
    shader->uploadAll(model,cam->view,cam->proj);

    buffer.bind();
    for(TextureGroup& tg : groups){
        shader->uploadTexture(tg.texture);

        int* start = 0 ;
        start += tg.startIndex;
        buffer.draw(tg.indices, (void*)start);
    }
     buffer.unbind();



    shader->unbind();
}

void TexturedAsset::renderDepth(Camera *cam, const mat4 &model)
{
    MVPTextureShader* shader = static_cast<MVPTextureShader*>(this->depthshader);

    shader->bind();
    shader->uploadAll(model,cam->view,cam->proj);

    buffer.bind();
    for(TextureGroup& tg : groups){
        shader->uploadTexture(tg.texture);

        int* start = 0 ;
        start += tg.startIndex;
        buffer.draw(tg.indices, (void*)start);
    }
     buffer.unbind();



    shader->unbind();
}


