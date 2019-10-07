/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */
#include "saiga/core/Core.h"
#include "saiga/core/image/ImageDraw.h"
#include "saiga/vision/orb/ORBextractor.h"

using namespace Saiga;


int main(int, char**)
{
    initSaigaSampleNoWindow();

    TemplatedImage<ucvec4> img("bar.png");
    TemplatedImage<unsigned char> imgGray(img.h, img.w);
    ImageTransformation::RGBAToGray8(img.getImageView(), imgGray.getImageView());
    int nfeatures = 1000;
    //    TemplatedImage<unsigned char> imgGray("test.png");

    Saiga::ORBextractor extractor(nfeatures, 1.2, 4, 20, 7);
    //    std::cout << img << std::endl;


    std::vector<Saiga::kpt_t> kps;
    Saiga::TemplatedImage<uchar> des;
    ivec2 bucketSize(80, 80);
    ivec2 dims(img.cols, img.rows);
    FeatureDistributionBucketing dis(dims, nfeatures, bucketSize);
    //FeatureDistributionSoftSSC dis(dims, nfeatures, 3, 0.1);
    //FeatureDistributionQuadtree dis(dims, nfeatures);
    //FeatureDistributionANMS dis(dims, nfeatures, FeatureDistributionANMS::AccelerationStructure::KDTREE);
    //FeatureDistributionANMS dis(dims, nfeatures, FeatureDistributionANMS::AccelerationStructure::RANGETREE);
    //FeatureDistributionANMS dis(dims, nfeatures, FeatureDistributionANMS::AccelerationStructure::GRID);
    //FeatureDistributionTopN dis(dims, nfeatures);

    for (int i = 0; i < 500; ++i)
    {
        kps.clear();
        SAIGA_BLOCK_TIMER("Extract");
        extractor(imgGray.getImageView(), kps, des, dis, true);
    }

    std::cout << "Found " << kps.size() << " keypoints." << std::endl;

    for (auto kp : kps)
    {
        ImageDraw::drawCircle(img.getImageView(), kp.point, 2, ucvec4(0, 0, 255, 255));
    }

    img.save("bar_features.png");


    return 0;
}
