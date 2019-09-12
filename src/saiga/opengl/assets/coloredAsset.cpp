/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "coloredAsset.h"

#include "saiga/opengl/shader/shaderLoader.h"
namespace Saiga
{
void ColoredAsset::loadDefaultShaders()
{
    this->shader          = shaderLoader.load<MVPShader>(deferredShaderStr);
    this->forwardShader   = shaderLoader.load<MVPShader>(forwardShaderStr);
    this->depthshader     = shaderLoader.load<MVPShader>(depthShaderStr);
    this->wireframeshader = shaderLoader.load<MVPShader>(wireframeShaderStr);
}
void LineVertexColoredAsset::loadDefaultShaders()
{
    this->shader          = shaderLoader.load<MVPShader>(deferredShaderStr);
    this->forwardShader   = shaderLoader.load<MVPShader>(forwardShaderStr);
    this->depthshader     = shaderLoader.load<MVPShader>(depthShaderStr);
    this->wireframeshader = shaderLoader.load<MVPShader>(wireframeShaderStr);
}
void TexturedAsset::loadDefaultShaders()
{
    this->shader          = shaderLoader.load<MVPShader>(deferredShaderStr);
    this->forwardShader   = shaderLoader.load<MVPShader>(forwardShaderStr);
    this->depthshader     = shaderLoader.load<MVPShader>(depthShaderStr);
    this->wireframeshader = shaderLoader.load<MVPShader>(wireframeShaderStr);
}
void TexturedAsset::render(Camera* cam, const mat4& model)
{
    auto tshader = std::static_pointer_cast<MVPTextureShader>(this->shader);
    tshader->bind();
    tshader->uploadModel(model);

    buffer.bind();
    for (TextureGroup& tg : groups)
    {
        tshader->uploadTexture(tg.texture.get());

        int start = 0;
        start += tg.startIndex;
        buffer.draw(tg.indices, start);
    }
    buffer.unbind();



    tshader->unbind();
}

void TexturedAsset::renderForward(Camera* cam, const mat4& model)
{
    render(cam, model);
}

void TexturedAsset::renderDepth(Camera* cam, const mat4& model)
{
    auto dshader = std::static_pointer_cast<MVPTextureShader>(this->depthshader);

    dshader->bind();
    dshader->uploadModel(model);

    buffer.bind();
    for (TextureGroup& tg : groups)
    {
        dshader->uploadTexture(tg.texture.get());

        int start = 0;
        start += tg.startIndex;
        buffer.draw(tg.indices, start);
    }
    buffer.unbind();



    dshader->unbind();
}



}  // namespace Saiga
