/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/query/gpuTimer.h"
#include "saiga/opengl/rendering/lighting/forward_lighting.h"
#include "saiga/opengl/rendering/renderer.h"
#include "saiga/opengl/rendering/deferredRendering/tone_mapper.h"

namespace Saiga
{
class ShaderLoader;


class SAIGA_OPENGL_API EmptyRenderer : public OpenGLRenderer
{
    class Asset;

   public:
    using ParameterType = RenderingParameters;
    ParameterType params;

    EmptyRenderer(OpenGLWindow& window, const ParameterType& params = ParameterType());
    virtual ~EmptyRenderer() {}

    virtual void renderGL(Framebuffer* target_framebuffer, ViewPort viewport, Camera* camera) override;
    virtual void renderImgui() override;

    void Resize(int windowWidth, int windowHeight);


    int renderWidth        = 0;
    int renderHeight       = 0;
};

}  // namespace Saiga
