/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/query/gpuTimer.h"
#include "saiga/opengl/rendering/renderer.h"
#include "saiga/opengl/rendering/standardForwardRendering/standardLighting/basic_forward_lighting.h"

namespace Saiga
{
class ShaderLoader;

struct SAIGA_OPENGL_API StandardForwardRenderingParameters : public RenderingParameters
{
    void fromConfigFile(const std::string& file) { RenderingParameters::fromConfigFile(file); }
};

class SAIGA_OPENGL_API StandardForwardRenderer : public OpenGLRenderer
{
   public:
    using InterfaceType = RenderingInterface;
    using ParameterType = StandardForwardRenderingParameters;

    ParameterType params;

    BasicForwardLighting lighting;

    StandardForwardRenderer(OpenGLWindow& window, const ParameterType& params = ParameterType());
    virtual ~StandardForwardRenderer() {}

    virtual void render(const RenderInfo& renderInfo) override;
    // void renderImGui(bool* p_open = nullptr) override;

    void resize(int width, int height);

    enum StandardForwardTimingBlock
    {
        TOTAL = 0,
        FORWARD,
        FINAL,
        COUNT,
    };

    float getBlockTime(StandardForwardTimingBlock timingBlock) { return timers[timingBlock].getTimeMS(); }
    virtual float getTotalRenderTime() override { return timers[StandardForwardTimingBlock::TOTAL].getTimeMS(); }

   private:
    std::vector<FilteredMultiFrameOpenGLTimer> timers;
    ShaderLoader shaderLoader;

   protected:
    void startTimer(StandardForwardTimingBlock timingBlock) { timers[timingBlock].startTimer(); }
    void stopTimer(StandardForwardTimingBlock timingBlock) { timers[timingBlock].stopTimer(); }
};

}  // namespace Saiga
