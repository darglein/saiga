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

struct SAIGA_OPENGL_API ForwardRenderingParameters : public RenderingParameters
{
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

   private:
    int renderWidth        = 0;
    int renderHeight       = 0;
    ShaderLoader shaderLoader;
    ToneMapper tone_mapper;

    std::shared_ptr<Texture> lightAccumulationTexture;
    Framebuffer lightAccumulationBuffer;

   protected:
    const char* coloredShaderSource  = "asset/ColoredAsset.glsl";
    const char* texturedShaderSource = "asset/texturedAsset.glsl";

    bool cullLights = false;

    bool depthPrepass = true;
};

}  // namespace Saiga
