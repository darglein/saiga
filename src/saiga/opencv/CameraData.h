#pragma once

#include "saiga/opencv/opencv.h"

#include "opencv2/core/core.hpp"

#include <opencv2/opencv.hpp>

namespace Saiga
{
struct SAIGA_GLOBAL Intrinsics
{
    int w, h;
    cv::Mat1d K, dist;

    void writeToFile(std::string file);
    void readFromFile(std::string file);
};

struct SAIGA_GLOBAL StereoExtrinsics
{
    cv::Mat1d R, t, E, F;

    cv::Matx44f getRelativeTransform();

    void writeToFile(std::string file);
    void readFromFile(std::string file);
};

}  // namespace Saiga
