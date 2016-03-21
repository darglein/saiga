#include "saiga/assets/coloredAsset.h"
#include "saiga/animation/boneShader.h"


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


void AnimatedAsset::render(Camera *cam, const mat4 &model)
{
    AnimationFrame af;
    Animation &a = animations[0];
    std::vector<mat4> boneMatrices = a.restPosition.boneMatrices;
//    cout<<"bonematrices "<<boneMatrices.size()<<endl;

    a.getFrame(test,af);
    af.calculateFromTree();
    boneMatrices = af.boneMatrices;

//    cout<<"bonematrices "<<boneMatrices.size()<<endl;

    test += 0.03f;


    BoneShader* shader = static_cast<BoneShader*>(this->shader);
    shader->bind();

//    std::vector<mat4> boneMatrices(boneCount);
    boneMatricesBuffer.updateBuffer(boneMatrices.data(),boneMatrices.size()*sizeof(mat4),0);
    boneMatricesBuffer.bind(shader->binding_boneMatricesBlock);

    shader->uploadAll(model,cam->view,cam->proj);
    buffer.bindAndDraw();
    shader->unbind();
}

void AnimatedAsset::renderDepth(Camera *cam, const mat4 &model)
{
    BoneShader* depthshader = static_cast<BoneShader*>(this->depthshader);
    depthshader->bind();
    depthshader->uploadAll(model,cam->view,cam->proj);
    buffer.bindAndDraw();
    depthshader->unbind();
}
