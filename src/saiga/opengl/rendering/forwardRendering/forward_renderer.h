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

class SAIGA_OPENGL_API Forward_Renderer : public OpenGLRenderer
{
   public:
    using InterfaceType = RenderingInterface;
    using ParameterType = ForwardRenderingParameters;

    ParameterType params;

    Forward_Renderer(OpenGLWindow& window, const ParameterType& params = ParameterType());
    virtual ~Forward_Renderer() {}

    virtual float getTotalRenderTime() override { return timer.getTimeMS(); }
    virtual void render(const RenderInfo& renderInfo) override;

   private:
    FilteredMultiFrameOpenGLTimer timer;
};

}  // namespace Saiga
