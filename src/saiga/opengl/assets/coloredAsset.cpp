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

ColoredAsset::ColoredAsset(const UnifiedModel& model)
{
    loadDefaultShaders();

    SAIGA_ASSERT(model.HasPosition());

    faces.reserve(model.NumFaces());
    for (auto& f : model.triangles)
    {
        faces.push_back({f(0), f(1), f(2)});
    }


    vertices.resize(model.NumVertices());
    for (int i = 0; i < model.NumVertices(); ++i)
    {
        vertices[i].position = make_vec4(model.position[i], 1);
    }

    if (model.HasColor())
    {
        for (int i = 0; i < model.NumVertices(); ++i)
        {
            vertices[i].color = model.color[i];
        }
    }
    else
    {
        for (int i = 0; i < model.NumVertices(); ++i)
        {
            vertices[i].color = vec4(1, 1, 1, 1);
        }
    }

    if (model.HasNormal())
    {
        for (int i = 0; i < model.NumVertices(); ++i)
        {
            vertices[i].normal = make_vec4(model.normal[i], 0);
        }
    }
    else
    {
        computePerVertexNormal();
    }

    std::cout << faces.size() << " " << vertices.size() << std::endl;
    create();
}
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
