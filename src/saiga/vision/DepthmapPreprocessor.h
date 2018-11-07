/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/image/image.h"


namespace Saiga {


struct SAIGA_GLOBAL DMPPParameters
{
    bool apply_downscale = false;
    int downscaleFactor = 2; // 2,4,8,16,...

    bool apply_filter = false;
    int filterRadius = 3;
    float sigmaFactor = 50.0f;
    int filterIterations = 1;

    bool apply_holeFilling = false;
    int holeFillIterations = 5;
    float fillDDscale = 0.5f;

    float dd_factor = 10.0f;



    /**
     *  Reads all paramters from the given config file.
     *  Creates the file with the default values if it doesn't exist.
     */
    void fromConfigFile(const std::string& file);


    void renderGui();
};



class SAIGA_GLOBAL DMPP
{
public:
    DMPPParameters params;
    DMPP(const DMPPParameters& params = DMPPParameters()) : params(params) {}


    void operator()(ImageView<float> src, ImageView<float> dst, glm::mat3 kinv);
    void renderGui();


private:
    void scaleDown2median(ImageView<float> src, ImageView<float> dst);
    void fillHoles(ImageView<float> vsrc, ImageView<float> vdst, glm::mat3 kinv);
    void applyFilterToImage(ImageView<float> vsrc, ImageView<float> vdst, glm::mat3 kinv);


    void computeMinMax(ImageView<float> vsrc, float& dmin, float& dmax);



};



}
