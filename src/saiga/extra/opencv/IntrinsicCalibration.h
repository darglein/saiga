#pragma once

#include "CalibrationPattern.h"
#include "CameraData.h"

namespace Saiga
{
class SAIGA_EXTRA_API IntrinsicsCalibration
{
   public:
    IntrinsicsCalibration(CalibrationPattern& pattern) : pattern(pattern) {}

    void addImage(cv::Mat image)
    {
        currentIntrinsics.w = image.cols;
        currentIntrinsics.h = image.rows;

        auto points = pattern.detect(image);
        if (points.size() > 0)
        {
            images.push_back(points);
        }
        else
        {
            std::cout << "could not find pattern :(" << std::endl;
        }
        recomputeIntrinsics();
    }

    void recomputeIntrinsics()
    {
        Intrinsics intr = currentIntrinsics;
        auto objPointss = pattern.duplicate(images.size());

        cv::Mat rvecs, tvecs;
        auto error = cv::calibrateCamera(objPointss, images, cv::Size(intr.w, intr.h), intr.K, intr.dist, rvecs, tvecs);
        std::cout << "calibrateCamera error: " << error << std::endl;

        currentIntrinsics = intr;
    }

    Intrinsics currentIntrinsics;

   protected:
    CalibrationPattern& pattern;
    std::vector<std::vector<CalibrationPattern::ImagePointType>> images;
};

}  // namespace Saiga
