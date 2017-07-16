#pragma once

#include "saiga/util/glm.h"
#include <vector>

namespace Saiga {

/**
 * Creation of "Lightning-like" line strips.
 *
 * Sources:
 * http://developer.download.nvidia.com/SDK/10/direct3d/Source/Lightning/doc/lightning_doc.pdf
 * http://drilian.com/2009/02/25/lightning-bolts/
 */

class SAIGA_GLOBAL Lightning{
public:
    struct SAIGA_GLOBAL LineSegment{
        float intensity;
        vec3 start, end;
        LineSegment(float i, vec3 s, vec3 e):intensity(i),start(s),end(e){}
    };

    static std::vector<LineSegment> createLightningBolt(vec3 startPoint, vec3 endPoint, int generations=5, float offsetAmount=1.0f, float splitProbability=0.4f, float splitLength=0.5f, float splitIntensityDrop=0.5f);
};

}
