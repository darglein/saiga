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
    this->deferredShader  = shaderLoader.load<MVPColorShader>(shaderStr, {{GL_FRAGMENT_SHADER, "#define DEFERRED", 1}});
    this->depthshader     = shaderLoader.load<MVPColorShader>(shaderStr, {{GL_FRAGMENT_SHADER, "#define DEPTH", 1}});
    this->forwardShader   = shaderLoader.load<MVPColorShader>(shaderStr);
    this->wireframeshader = shaderLoader.load<MVPColorShader>(shaderStr);
}

ColoredAsset::ColoredAsset(const TriangleMesh<VertexNC, uint32_t>& mesh)
{
    this->vertices = mesh.vertices;
    this->faces    = mesh.faces;
    create();
}

ColoredAsset::ColoredAsset(const UnifiedModel& model) : ColoredAsset(model.Mesh<VertexNC, uint32_t>()) {}

void LineVertexColoredAsset::loadDefaultShaders()
{
    this->deferredShader  = shaderLoader.load<MVPColorShader>(shaderStr, {{GL_FRAGMENT_SHADER, "#define DEFERRED", 1}});
    this->depthshader     = shaderLoader.load<MVPColorShader>(shaderStr, {{GL_FRAGMENT_SHADER, "#define DEPTH", 1}});
    this->forwardShader   = shaderLoader.load<MVPColorShader>(shaderStr);
    this->wireframeshader = shaderLoader.load<MVPColorShader>(shaderStr);
}

void LineVertexColoredAsset::SetShaderColor(const vec4& color)
{
    deferredShader->bind();
    deferredShader->uploadColor(color);
    deferredShader->unbind();

    forwardShader->bind();
    forwardShader->uploadColor(color);
    forwardShader->unbind();
}

TexturedAsset::TexturedAsset(const UnifiedModel& model)
{
    this->TriangleMesh<VertexNTD, uint32_t>::operator=(model.Mesh<VertexNTD, uint32_t>());
    create();


    for (auto& mg : model.material_groups)
    {
        auto& material = model.materials[mg.materialId];

        TexturedAsset::TextureGroup tg;
        tg.startIndex = mg.startFace * 3;
        tg.indices    = mg.numFaces * 3;
        tg.texture    = TextureLoader::instance()->load(material.texture_diffuse);
        if (tg.texture)
        {
            tg.texture->setWrap(GL_REPEAT);
            tg.texture->generateMipmaps();
            groups.push_back(tg);
        }
    }
}


void TexturedAsset::loadDefaultShaders()
{
    this->deferredShader =
        shaderLoader.load<MVPTextureShader>(shaderStr, {{GL_FRAGMENT_SHADER, "#define DEFERRED", 1}});
    this->depthshader     = shaderLoader.load<MVPTextureShader>(shaderStr, {{GL_FRAGMENT_SHADER, "#define DEPTH", 1}});
    this->forwardShader   = shaderLoader.load<MVPTextureShader>(shaderStr);
    this->wireframeshader = shaderLoader.load<MVPTextureShader>(shaderStr);
}


void TexturedAsset::render(Camera* cam, const mat4& model)
{
    renderGroups(deferredShader, cam, model);
}

void TexturedAsset::renderForward(Camera* cam, const mat4& model)
{
    renderGroups(forwardShader, cam, model);
}

void TexturedAsset::renderDepth(Camera* cam, const mat4& model)
{
    renderGroups(depthshader, cam, model);
}

void TexturedAsset::renderGroups(std::shared_ptr<MVPTextureShader> shader, Camera* cam, const mat4& model)
{
    shader->bind();
    shader->uploadModel(model);
    buffer.bind();
    for (TextureGroup& tg : groups)
    {
        shader->uploadTexture(tg.texture.get());
        int start = 0;
        start += tg.startIndex;
        buffer.draw(tg.indices, start);
    }
    buffer.unbind();
    shader->unbind();
}



}  // namespace Saiga
