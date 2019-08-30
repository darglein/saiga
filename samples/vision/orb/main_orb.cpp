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
    Saiga::SaigaParameters saigaParameters;
    Saiga::initSample(saigaParameters);
    Saiga::initSaiga(saigaParameters);

    TemplatedImage<ucvec4> img("bar.png");
    TemplatedImage<unsigned char> imgGray(img.h, img.w);
    ImageTransformation::RGBAToGray8(img.getImageView(), imgGray.getImageView());
    //    TemplatedImage<unsigned char> imgGray("test.png");

    SaigaORB::ORBextractor extractor(1000, 1.2, 4, 20, 7);
    //    std::cout << img << std::endl;


    std::vector<SaigaORB::kpt_t> kps;
    Saiga::TemplatedImage<uchar> des;

    for (int i = 0; i < 5; ++i)
    {
        kps.clear();
        SAIGA_BLOCK_TIMER("Extract");
        extractor(imgGray.getImageView(), kps, des, true);
    }

    std::cout << "Found " << kps.size() << " keypoints." << std::endl;

    for (auto kp : kps)
    {
        ImageDraw::drawCircle(img.getImageView(), kp.point, 2, ucvec4(0, 0, 255, 255));
    }

    img.save("bar_features.png");


    return 0;
}
