/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/opengl/rendering/overlay/deferredDebugOverlay.h"

#include "saiga/core/geometry/triangle_mesh.h"
#include "saiga/core/geometry/triangle_mesh_generator.h"
#include "saiga/opengl/framebuffer.h"
#include "saiga/opengl/rendering/deferredRendering/gbuffer.h"
#include "saiga/opengl/shader/basic_shaders.h"
#include "saiga/opengl/shader/shaderLoader.h"


namespace Saiga
{
DeferredDebugOverlay::DeferredDebugOverlay(int width, int height) : layout(width, height)
{
    auto tm = TriangleMeshGenerator::createFullScreenQuadMesh();

    float aspect = float(width) / height;
    tm->transform(scale(vec3(aspect, 1, 1)));

    meshBB = tm->aabb();

    buffer.fromMesh(*tm);


    setScreenPosition(&color, 0);
    setScreenPosition(&normal, 1);
    setScreenPosition(&depth, 2);
    setScreenPosition(&data, 3);
    setScreenPosition(&light, 4);
}

void DeferredDebugOverlay::loadShaders()
{
    shader       = shaderLoader.load<MVPTextureShader>("debug/gbuffer.glsl");
    depthShader  = shaderLoader.load<MVPTextureShader>("debug/gbuffer_depth.glsl");
    normalShader = shaderLoader.load<MVPTextureShader>("debug/gbuffer_normal.glsl");
}

void DeferredDebugOverlay::setScreenPosition(GbufferTexture* gbt, int id)
{
    float images = 5;

    float s = 1.0f / images;

    layout.transform(gbt, meshBB, vec2(1, 1 - s * id), s, Layout::RIGHT, Layout::RIGHT);
    return;
    gbt->setScale(make_vec3(s));
    float dy = -s * 2.0f;
    float y  = id * dy + dy * 0.5f + 1.0f;
    gbt->translateGlobal(vec3(1.0f - s, y, 0));
    gbt->calculateModel();
}

void DeferredDebugOverlay::render()
{
    // lazy shader loading
    if (!shader) loadShaders();


    glDisable(GL_DEPTH_TEST);
    glDisable(GL_BLEND);
    shader->bind();

    shader->uploadModel(color.model);
    shader->uploadTexture(color.texture.get());
    buffer.bindAndDraw();

    shader->uploadModel(data.model);
    shader->uploadTexture(data.texture.get());
    buffer.bindAndDraw();

    shader->uploadModel(light.model);
    shader->uploadTexture(light.texture.get());
    buffer.bindAndDraw();

    shader->unbind();


    normalShader->bind();
    normalShader->uploadModel(normal.model);
    normalShader->uploadTexture(normal.texture.get());
    buffer.bindAndDraw();
    normalShader->unbind();

    depthShader->bind();

    depthShader->uploadModel(depth.model);
    depthShader->uploadTexture(depth.texture.get());
    buffer.bindAndDraw();

    depthShader->unbind();
}

void DeferredDebugOverlay::setDeferredFramebuffer(GBuffer* gbuffer, std::shared_ptr<TextureBase> light)
{
    color.texture       = gbuffer->getTextureColor();
    normal.texture      = gbuffer->getTextureNormal();
    depth.texture       = gbuffer->getTextureDepth();
    data.texture        = gbuffer->getTextureData();
    this->light.texture = light;
}

}  // namespace Saiga
