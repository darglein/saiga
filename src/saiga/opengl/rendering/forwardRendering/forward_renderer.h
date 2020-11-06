/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/query/gpuTimer.h"
#include "saiga/opengl/rendering/forwardRendering/forwardLighting/forward_lighting.h"
#include "saiga/opengl/rendering/renderer.h"

namespace Saiga
{
class ShaderLoader;

struct SAIGA_OPENGL_API ForwardRenderingParameters : public RenderingParameters
{
    void fromConfigFile(const std::string& file) { RenderingParameters::fromConfigFile(file); }
};

class SAIGA_OPENGL_API ForwardRenderer : public OpenGLRenderer
{
    class Asset;

   public:
    using InterfaceType = RenderingInterface;
    using ParameterType = ForwardRenderingParameters;

    ParameterType params;

    ForwardLighting lighting;

    ForwardRenderer(OpenGLWindow& window, const ParameterType& params = ParameterType());
    virtual ~ForwardRenderer() {}

    virtual void render(const RenderInfo& renderInfo) override;
    // void renderImGui(bool* p_open = nullptr) override;

    void resize(int width, int height);

    inline const char* getMainShaderSource()
    {
        return mainShaderSource;
    }

    enum ForwardTimingBlock
    {
        TOTAL = 0,
        FORWARD,
        FINAL,
        COUNT,
    };

    float getBlockTime(ForwardTimingBlock timingBlock) { return timers[timingBlock].getTimeMS(); }
    virtual float getTotalRenderTime() override { return timers[ForwardTimingBlock::TOTAL].getTimeMS(); }

   private:
    std::vector<FilteredMultiFrameOpenGLTimer> timers;
    ShaderLoader shaderLoader;

   protected:
    void startTimer(ForwardTimingBlock timingBlock) { timers[timingBlock].startTimer(); }
    void stopTimer(ForwardTimingBlock timingBlock) { timers[timingBlock].stopTimer(); }

    const char* mainShaderSource = "asset/forwardColoredAsset.glsl";
};

}  // namespace Saiga
