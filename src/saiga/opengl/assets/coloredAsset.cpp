/**
 * Copyright (c) 2021 Darius Rückert
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

ColoredAsset::ColoredAsset(const UnifiedMesh& model) : ColoredAsset()
{
    SAIGA_ASSERT(model.HasColor());
    unified_buffer = std::make_shared<UnifiedMeshBuffer>(model);
}


ColoredAsset::ColoredAsset(const UnifiedModel& model) : ColoredAsset()
{
    auto [mesh, groups] = model.CombinedMesh(VERTEX_POSITION | VERTEX_NORMAL | VERTEX_COLOR);
    this->groups        = groups;

    unified_buffer = std::make_shared<UnifiedMeshBuffer>(mesh);
}


void LineVertexColoredAsset::loadDefaultShaders()
{
    this->deferredShader  = shaderLoader.load<MVPColorShader>(shaderStr, {{GL_FRAGMENT_SHADER, "#define DEFERRED", 1}});
    this->depthshader     = shaderLoader.load<MVPColorShader>(shaderStr, {{GL_FRAGMENT_SHADER, "#define DEPTH", 1}});
    this->forwardShader   = shaderLoader.load<MVPColorShader>(shaderStr);
    this->wireframeshader = shaderLoader.load<MVPColorShader>(shaderStr);
}


LineVertexColoredAsset::LineVertexColoredAsset(const UnifiedMesh& model) : LineVertexColoredAsset()
{
    unified_buffer = std::make_shared<UnifiedMeshBuffer>(model, GL_LINES);
}

void LineVertexColoredAsset::SetShaderColor(const vec4& color)
{
    if (deferredShader->bind())
    {
        deferredShader->uploadColor(color);
        deferredShader->unbind();
    }

    if (forwardShader->bind())
    {
        forwardShader->uploadColor(color);
        forwardShader->unbind();
    }
}

void LineVertexColoredAsset::SetRenderFlags(RenderFlags flags)
{
    if (deferredShader->bind())
    {
        deferredShader->upload(3, int(flags));
        deferredShader->unbind();
    }

    if (forwardShader->bind())
    {
        forwardShader->upload(3, int(flags));
        forwardShader->unbind();
    }
}

TexturedAsset::TexturedAsset(const UnifiedModel& model) : TexturedAsset()
{
    auto [mesh, groups] = model.CombinedMesh(VERTEX_POSITION | VERTEX_NORMAL | VERTEX_TEXTURE_COORDINATES);
    this->groups        = groups;

    unified_buffer = std::make_shared<UnifiedMeshBuffer>(mesh);

    texture_name_to_id = model.texture_name_to_id;

    materials = model.materials;
    for (auto& tex : model.textures)
    {
        auto texture = std::make_shared<Texture>(tex, false, true);
        texture->setWrap(GL_REPEAT);
        texture->generateMipmaps();
        textures.push_back(texture);
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
    if (shader->bind())
    {
        shader->uploadModel(model);
        RenderNoShaderBind(shader.get());
        shader->unbind();
    }
}
void TexturedAsset::RenderNoShaderBind(MVPTextureShader* shader)
{
    unified_buffer->Bind();
    for (int i = 0; i < groups.size(); ++i)
    {
        auto& tg = groups[i];
        SAIGA_ASSERT(tg.materialId >= 0 && tg.materialId < materials.size());

        auto& mat = materials[tg.materialId];

        SAIGA_ASSERT(!mat.texture_diffuse.empty());
        if (mat.texture_diffuse.empty()) continue;

        SAIGA_ASSERT(!mat.texture_diffuse.empty());
        auto& tex = textures[texture_name_to_id[mat.texture_diffuse]];

        SAIGA_ASSERT(tex);
        if (tex)
        {
            shader->uploadTexture(tex.get());
            unified_buffer->Draw(tg.startFace, tg.numFaces);
        }
    }
    unified_buffer->Unbind();
}



}  // namespace Saiga
