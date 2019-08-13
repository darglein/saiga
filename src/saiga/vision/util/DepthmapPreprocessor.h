/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/core/image/image.h"
#include "saiga/vision/VisionTypes.h"

namespace Saiga
{
struct SAIGA_VISION_API DMPPParameters
{
    bool apply_downscale = false;
    int downscaleFactor  = 2;  // 2,4,8,16,...

    bool apply_filter    = false;
    int filterRadius     = 3;
    float sigmaFactor    = 50.0f;
    int filterIterations = 1;

    bool apply_holeFilling = false;
    int holeFillIterations = 5;
    float fillDDscale      = 0.5f;

    float dd_factor = 10.0f;



    /**
     *  Reads all paramters from the given config file.
     *  Creates the file with the default values if it doesn't exist.
     */
    void fromConfigFile(const std::string& file);


    void renderGui();
};



class SAIGA_VISION_API DMPP
{
   public:
    using DepthMap = ImageView<float>;


    DMPPParameters params;
    DMPP(const Intrinsics4& camera = Intrinsics4(), const DMPPParameters& params = DMPPParameters());


    void operator()(DepthMap src, DepthMap dst);
    // Inplace preprocessing
    void operator()(DepthMap src);

    void setCamera(const Intrinsics4& c) { camera = c; }

    void scaleDown2median(DepthMap src, DepthMap dst);
    void fillHoles(DepthMap vsrc, DepthMap vdst);
    void applyFilterToImage(DepthMap vsrc, DepthMap vdst);
    void computeMinMax(DepthMap vsrc, float& dmin, float& dmax);

    void renderGui();

   private:
    Intrinsics4 camera;
};



}  // namespace Saiga
