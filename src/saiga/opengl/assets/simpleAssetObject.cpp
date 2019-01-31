/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */


#include "simpleAssetObject.h"

#include "saiga/camera/camera.h"
#include "saiga/opengl/assets/asset.h"

namespace Saiga
{
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
