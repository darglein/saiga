/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "saiga/util/lightning.h"

#include "internal/noGraphicsAPI.h"

namespace Saiga
{
std::vector<Lightning::LineSegment> Lightning::createLightningBolt(vec3 startPoint, vec3 endPoint, int generations,
                                                                   float offsetAmount, float splitProbability,
                                                                   float splitLength, float splitIntensityDrop)
{
    vec3 direction = normalize(endPoint - startPoint);
    vec3 right     = normalize(cross(direction, vec3(0, 1, 0)));
    vec3 up        = normalize(cross(direction, right));

    std::vector<LineSegment> segmentList;
    segmentList.emplace_back(1.0f, startPoint, endPoint);

    //    float offsetAmount = 1.0f; // the maximum amount to offset a lightning vertex.
    for (int gen = 0; gen < generations; ++gen)
    {
        //        cout<<"gen "<<gen<<endl;
        int segCount = segmentList.size();
        //      for each segment that was in segmentList when this generation started
        for (int seg = 0; seg < segCount; ++seg)
        {
            //        segmentList.Remove(segment); // This segment is no longer necessary.
            LineSegment& s = segmentList[seg];
            //            cout<<"seg "<<seg<<" "<<s.start<<" "<<s.end<<endl;
            vec3 midPoint = (s.start + s.end) * 0.5f;
            // Offset the midpoint by a random amount along the normal.
            vec2 offset = glm::diskRand(1.0f);
            midPoint += (right * offset[0] + up * offset.y) * offsetAmount;



            // Create two new segments that span from the start point to the end point,
            // but with the new (randomly-offset) midpoint.
            float i = s.intensity;
            vec3 e  = s.end;
            vec3 st = s.start;
            s.end   = midPoint;
            segmentList.emplace_back(i, midPoint, e);

            if (linearRand(0.0f, 1.0f) < splitProbability)
            {
                vec3 currentDir = midPoint - st;
                vec3 splitEnd   = currentDir * splitLength +
                                midPoint;  // lengthScale is, for best results, < 1.  0.7 is a good value.
                segmentList.emplace_back(i * splitIntensityDrop, midPoint, splitEnd);
            }
        }
        offsetAmount /= 2;  // Each subsequent generation offsets at max half as much as the generation before.
    }

    return segmentList;
}

}  // namespace Saiga
