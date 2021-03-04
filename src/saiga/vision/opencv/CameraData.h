#pragma once

#include "saiga/vision/opencv/opencv.h"

#include "opencv2/core/core.hpp"

#include <opencv2/opencv.hpp>

namespace Saiga
{
struct SAIGA_EXTRA_API Intrinsics
{
    int w, h;
    cv::Mat1d K, dist;

    void writeToFile(std::string file)
    {
        cv::FileStorage fs(file, FileStorage::WRITE);

        fs << "w" << w;
        fs << "h" << h;
        fs << "K" << K;
        fs << "dist" << dist;

        std::cout << "Saved Intrinsics to " << file << std::endl;
    }
    bool readFromFile(std::string file)
    {
        cv::FileStorage fs(file, FileStorage::READ);
        if (!fs.isOpened()) return false;
        fs["w"] >> w;
        fs["h"] >> h;
        fs["K"] >> K;
        fs["dist"] >> dist;

        std::cout << "Loaded Intrinsics from " << file << std::endl;
        return true;
    }
};

struct SAIGA_EXTRA_API StereoExtrinsics
{
    cv::Mat1d R, t, E, F;

    cv::Matx44f getRelativeTransform()
    {
        Matx44f M = Matx44f::eye();
        for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j) M(i, j) = R(i, j);
        for (int i = 0; i < 3; ++i) M(i, 3) = t(i);
        return M;
    }

    void writeToFile(std::string file)
    {
        cv::FileStorage fs(file, FileStorage::WRITE);

        fs << "R" << R;
        fs << "t" << t;
        fs << "F" << F;
        fs << "E" << E;

        std::cout << "Saved StereoExtrinsics to " << file << std::endl;
    }
    bool readFromFile(std::string file)
    {
        cv::FileStorage fs(file, FileStorage::READ);
        if (!fs.isOpened()) return false;
        fs["R"] >> R;
        fs["t"] >> t;
        fs["F"] >> F;
        fs["E"] >> E;

        std::cout << "Loaded StereoExtrinsics from " << file << std::endl;
        return true;
    }
};

}  // namespace Saiga
