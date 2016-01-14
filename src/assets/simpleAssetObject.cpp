#include "saiga/assets/simpleAssetObject.h"

#include "saiga/assets/asset.h"
#include "saiga/camera/camera.h"

void SimpleAssetObject::render(Camera *cam)
{
    asset->render(cam,model);
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
