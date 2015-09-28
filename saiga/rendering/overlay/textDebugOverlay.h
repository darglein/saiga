#pragma once

#include "saiga/util/glm.h"
#include "saiga/opengl/indexedVertexBuffer.h"
#include "saiga/rendering/object3d.h"

#include "saiga/text/TextOverlay2D.h"
#include <vector>


class TextGenerator;

class DynamicText;

class SAIGA_GLOBAL TextDebugOverlay {
public:
    TextOverlay2D overlay;
    TextGenerator* textGenerator;

    DynamicText* text;

    TextDebugOverlay();

    void init(TextGenerator* textGenerator);
    void render();



};


