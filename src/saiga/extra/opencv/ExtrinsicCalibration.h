#pragma once

#include "CalibrationPattern.h"
#include "CameraData.h"

namespace Saiga
{
class SAIGA_EXTRA_API StereoCalibration
{
   public:
    StereoCalibration(CalibrationPattern& pattern, Intrinsics intr1, Intrinsics intr2)
        : pattern(pattern), intr1(intr1), intr2(intr2)
    {
    }

    void addImage(cv::Mat image1, cv::Mat image2)
    {
        auto points1 = pattern.detect(image1);
        auto points2 = pattern.detect(image2);

        if (points1.size() > 0 && points2.size() > 0)
        {
            images1.push_back(points1);
            images2.push_back(points2);
            recomputeExtrinsics();
        }
    }

    void recomputeExtrinsics()
    {
        SAIGA_ASSERT(pattern.objPoints.size() > 0);
        SAIGA_ASSERT(images1.size() > 0);

        auto objPointss = pattern.duplicate(images1.size());



        auto error = cv::stereoCalibrate(objPointss, images1, images2, intr1.K, intr1.dist, intr2.K, intr2.dist,
                                         cv::Size(0, 0), extr.R, extr.t, extr.E, extr.F, cv::CALIB_FIX_INTRINSIC);

        std::cout << "stereoCalibrate error: " << error << std::endl;
    }

    StereoExtrinsics extr;

   protected:
    CalibrationPattern& pattern;

    Intrinsics intr1;
    Intrinsics intr2;


    std::vector<std::vector<CalibrationPattern::ImagePointType>> images1;
    std::vector<std::vector<CalibrationPattern::ImagePointType>> images2;
};
}  // namespace Saiga
