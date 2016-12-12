#include "saiga/assets/coloredAsset.h"


void TexturedAsset::render(Camera *cam, const mat4 &model)
{
    MVPTextureShader* tshader = static_cast<MVPTextureShader*>(this->shader);
    tshader->bind();
    tshader->uploadModel(model);

    buffer.bind();
    for(TextureGroup& tg : groups){
		tshader->uploadTexture(tg.texture);

        int start = 0 ;
        start += tg.startIndex;
        buffer.draw(tg.indices, start);
    }
     buffer.unbind();



	 tshader->unbind();
}

void TexturedAsset::renderDepth(Camera *cam, const mat4 &model)
{
    MVPTextureShader* dshader = static_cast<MVPTextureShader*>(this->depthshader);

    dshader->bind();
    dshader->uploadModel(model);

    buffer.bind();
    for(TextureGroup& tg : groups){
		dshader->uploadTexture(tg.texture);

        int start = 0 ;
        start += tg.startIndex;
        buffer.draw(tg.indices, start);
    }
     buffer.unbind();



	 dshader->unbind();
}


