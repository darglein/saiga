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

LineVertexColoredAsset::LineVertexColoredAsset(const LineMesh<VertexNC, uint32_t>& line_mesh)
{
    this->vertices = line_mesh.vertices;
    this->lines    = line_mesh.lines;
    create();
}

LineVertexColoredAsset::LineVertexColoredAsset(const UnifiedModel& model)
    : LineVertexColoredAsset(model.LineMesh<VertexNC, uint32_t>())
{
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

void LineVertexColoredAsset::SetRenderFlags(RenderFlags flags)
{
    deferredShader->bind();
    deferredShader->upload(3, int(flags));
    deferredShader->unbind();

    forwardShader->bind();
    forwardShader->upload(3, int(flags));
    forwardShader->unbind();
}

TexturedAsset::TexturedAsset(const UnifiedModel& model) : groups(model.material_groups), materials(model.materials)
{
    this->TriangleMesh<VertexNTD, uint32_t>::operator=(model.Mesh<VertexNTD, uint32_t>());
    create();


    for (auto& material : model.materials)
    {
        if (material.texture_diffuse.empty())
        {
            TemplatedImage<ucvec4> img(10, 10);
            img.getImageView().set(ucvec4(100, 100, 100, 255));
            auto tex = std::make_shared<Texture>(img);
            textures.push_back(tex);
            continue;
        }

        auto texture = TextureLoader::instance()->load(material.texture_diffuse);
        if (texture)
        {
            texture->setWrap(GL_REPEAT);
            texture->generateMipmaps();
            textures.push_back(texture);
        }
        else
        {
            throw std::runtime_error("Could not load texture " + material.texture_diffuse);
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
    for (auto& tg : groups)
    {
        auto& tex = textures[tg.materialId];
        if (tex)
        {
            shader->uploadTexture(tex.get());
            buffer.draw(tg.numFaces * 3, tg.startFace * 3);
        }
    }
    buffer.unbind();
    shader->unbind();
}



}  // namespace Saiga
