/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/assets/simpleAssetObject.h"

#include "saiga/assets/asset.h"
#include "saiga/camera/camera.h"

namespace Saiga {

void SimpleAssetObject::render(Camera *cam)
{
    asset->render(cam,model);
}

void SimpleAssetObject::renderForward(Camera *cam)
{
    asset->renderForward(cam,model);
}

void SimpleAssetObject::renderDepth(Camera *cam)
{
    asset->renderDepth(cam,model);
}

void SimpleAssetObject::renderWireframe(Camera *cam)
{
    asset->renderWireframe(cam,model);
}

void SimpleAssetObject::renderRaw()
{
    asset->renderRaw();
}

}
