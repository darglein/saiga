/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/config.h"
#include "saiga/core/geometry/object3d.h"
#include "saiga/core/window/Interfaces.h"

#include <memory>

namespace Saiga
{
class Asset;
class Camera;

class SAIGA_OPENGL_API SimpleAssetObject : public Object3D
{
   public:
    std::shared_ptr<Asset> asset;

    void render(Camera* camera, RenderPass render_pass);

    void render(Camera* cam);
    void renderForward(Camera* cam);
    void renderDepth(Camera* cam);
    void renderWireframe(Camera* cam);
    void renderRaw();
};

}  // namespace Saiga
