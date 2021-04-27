/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "simpleAssetObject.h"

#include "saiga/core/camera/camera.h"
#include "saiga/opengl/assets/asset.h"

namespace Saiga
{
void SimpleAssetObject::render(Camera* camera, RenderPass render_pass)
{
    switch (render_pass)
    {
        case RenderPass::Forward:
            renderForward(camera);
            break;
        case RenderPass::Deferred:
            render(camera);
            break;
        case RenderPass::Shadow:
        case RenderPass::DepthPrepass:
            renderDepth(camera);
            break;
        default:
            break;
    }
}

void SimpleAssetObject::render(Camera* cam)
{
    SAIGA_ASSERT(asset);
    asset->render(cam, model);
}

void SimpleAssetObject::renderForward(Camera* cam)
{
    SAIGA_ASSERT(asset);
    asset->renderForward(cam, model);
}

void SimpleAssetObject::renderDepth(Camera* cam)
{
    SAIGA_ASSERT(asset);
    asset->renderDepth(cam, model);
}

void SimpleAssetObject::renderWireframe(Camera* cam)
{
    SAIGA_ASSERT(asset);
    asset->renderWireframe(cam, model);
}

void SimpleAssetObject::renderRaw()
{
    SAIGA_ASSERT(asset);
    asset->renderRaw();
}

}  // namespace Saiga
