/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/query/gpuTimer.h"
#include "saiga/opengl/rendering/renderer.h"

namespace Saiga
{
struct SAIGA_OPENGL_API ForwardRenderingParameters : public RenderingParameters
{
    void fromConfigFile(const std::string& file) { RenderingParameters::fromConfigFile(file); }
};

class SAIGA_OPENGL_API ForwardRenderer : public OpenGLRenderer
{
   public:
    using ParameterType = ForwardRenderingParameters;

    ParameterType params;

    ForwardRenderer(OpenGLWindow& window, const ParameterType& params = ParameterType());
    virtual ~ForwardRenderer() {}

    virtual float getTotalRenderTime() override { return timer.getTimeMS(); }
    virtual void render(const RenderInfo& renderInfo) override;

   private:
    FilteredMultiFrameOpenGLTimer timer;
};

}  // namespace Saiga
