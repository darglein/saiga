/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"

struct ImDrawData;

namespace Saiga {

class SAIGA_GLOBAL ImGuiRenderer{
public:
    bool wantsCaptureMouse = false;

    virtual ~ImGuiRenderer(){}

    void checkWindowFocus();

    virtual void shutdown() = 0;
    virtual void beginFrame() = 0;
    virtual void endFrame();
    virtual void renderDrawLists(ImDrawData *draw_data) = 0;
};

}
