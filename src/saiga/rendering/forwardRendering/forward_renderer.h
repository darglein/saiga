/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/rendering/renderer.h"
#include "saiga/opengl/query/gpuTimer.h"

namespace Saiga {


class SAIGA_GLOBAL ForwardRenderingInterface : public RenderingBase
{
public:
    ForwardRenderingInterface(RendererBase& parent) : RenderingBase(parent) {}
    virtual ~ForwardRenderingInterface(){}

    //forward rendering path after lighting, but before post processing
    //this could be used for transparent objects
    virtual void renderOverlay(Camera *cam) {}

    //forward rendering path after lighting and after post processing
    virtual void renderFinal(Camera *cam) {}
//protected:
//    RendererBase& parentRenderer;
};

struct SAIGA_GLOBAL ForwardRenderingParameters : public RenderingParameters
{


};

class SAIGA_GLOBAL Forward_Renderer : public Renderer{

public:
    ForwardRenderingParameters params;

    Forward_Renderer(OpenGLWindow& window, const ForwardRenderingParameters& params = ForwardRenderingParameters());
    virtual ~Forward_Renderer() {}

    virtual float getTotalRenderTime() override { return timer.getTimeMS(); }
    virtual void render(Camera *cam) override;
private:
    FilteredMultiFrameOpenGLTimer timer;
};

}
