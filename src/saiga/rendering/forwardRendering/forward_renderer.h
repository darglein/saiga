/**
 * Copyright (c) 2017 Darius Rückert 
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/rendering/renderer.h"
#include "saiga/opengl/query/gpuTimer.h"

namespace Saiga {



class SAIGA_GLOBAL Forward_Renderer : public Renderer{

public:


    Forward_Renderer(OpenGLWindow& window);
    virtual ~Forward_Renderer() {}

    virtual float getTotalRenderTime() override { return timer.getTimeMS(); }
    virtual void render_intern(Camera *cam) override;
private:
    FilteredMultiFrameOpenGLTimer timer;
};

}
