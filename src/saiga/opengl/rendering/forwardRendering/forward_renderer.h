/**
 * Copyright (c) 2020 Paul Himmler
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/opengl/query/gpuTimer.h"
#include "saiga/opengl/rendering/lighting/forward_lighting.h"
#include "saiga/opengl/rendering/renderer.h"

namespace Saiga
{
class ShaderLoader;

struct SAIGA_OPENGL_API ForwardRenderingParameters : public RenderingParameters
{
    int maximumNumberOfDirectionalLights = 256;
    int maximumNumberOfPointLights       = 256;
    int maximumNumberOfSpotLights        = 256;


    void fromConfigFile(const std::string& file) { RenderingParameters::fromConfigFile(file); }
};

class SAIGA_OPENGL_API ForwardRenderer : public OpenGLRenderer
{
    class Asset;

   public:
    using ParameterType = ForwardRenderingParameters;


    ParameterType params;
    ForwardLighting lighting;

    ForwardRenderer(OpenGLWindow& window, const ParameterType& params = ParameterType());
    virtual ~ForwardRenderer() {}

    virtual void renderGL(Framebuffer* target_framebuffer, ViewPort viewport, Camera* camera) override;
    virtual void renderImgui() override;

    void Resize(int windowWidth, int windowHeight);

    inline const char* getColoredShaderSource() { return coloredShaderSource; }
    inline const char* getTexturedShaderSource() { return texturedShaderSource; }

    enum ForwardTimingBlock
    {
        TOTAL = 0,
        FORWARD,
        FINAL,
        COUNT,
    };

    float getBlockTime(ForwardTimingBlock timingBlock) { return timers[timingBlock].getTimeMS(); }
    virtual float getTotalRenderTime() override { return timers[ForwardTimingBlock::TOTAL].getTimeMS(); }

    inline void setLightMaxima(int maxDirectionalLights, int maxPointLights, int maxSpotLights)
    {
        params.maximumNumberOfDirectionalLights = maxDirectionalLights;
        params.maximumNumberOfPointLights       = maxPointLights;
        params.maximumNumberOfSpotLights        = maxSpotLights;

        params.maximumNumberOfDirectionalLights = std::max(0, params.maximumNumberOfDirectionalLights);
        params.maximumNumberOfPointLights       = std::max(0, params.maximumNumberOfPointLights);
        params.maximumNumberOfSpotLights        = std::max(0, params.maximumNumberOfSpotLights);

        lighting.setLightMaxima(params.maximumNumberOfDirectionalLights, params.maximumNumberOfPointLights,
                                params.maximumNumberOfSpotLights);
    }

   private:
    int renderWidth  = 0;
    int renderHeight = 0;
    std::vector<FilteredMultiFrameOpenGLTimer> timers;
    bool showLightingImgui = false;
    ShaderLoader shaderLoader;

   protected:
    void startTimer(ForwardTimingBlock timingBlock) { timers[timingBlock].startTimer(); }
    void stopTimer(ForwardTimingBlock timingBlock) { timers[timingBlock].stopTimer(); }

    const char* coloredShaderSource  = "asset/ColoredAsset.glsl";
    const char* texturedShaderSource = "asset/TexturedAsset.glsl";

    bool cullLights = false;

    bool depthPrepass = true;
};

}  // namespace Saiga
