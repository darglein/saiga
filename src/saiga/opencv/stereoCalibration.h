#pragma once

#include "saiga/opencv/opencv.h"

#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"


struct SAIGA_GLOBAL Intrinsics
{
    int w, h;
    cv::Mat K, dist;

    void writeToFile(std::string file);
    void readFromFile(std::string file);
};

struct SAIGA_GLOBAL StereoExtrinsics
{
    cv::Mat R,t,E,F;

    void writeToFile(std::string file);
    void readFromFile(std::string file);
};

class SAIGA_GLOBAL StereoCalibration
{
public:
    enum class CalibrationPattern
    {
        CHESSBOARD
    };



    StereoCalibration(
            cv::Size patternSize = cv::Size(13,6),
            double patternMetricDistance = 0.03, // 3cm
            CalibrationPattern pattern = CalibrationPattern::CHESSBOARD
            );


    Intrinsics calibrateIntrinsics(std::vector<cv::Mat> images);

private:
    std::vector<cv::Point2f> findPattern(cv::Mat image);

    Intrinsics calibrateIntrinsics(std::vector<std::vector<cv::Point2f>>& corners, int w, int h);

    StereoExtrinsics calibrateStereo(
            Intrinsics intrinsics1, Intrinsics intrinsics2,
            std::vector<std::vector<cv::Point2f>>& corners1,
            std::vector<std::vector<cv::Point2f>>& corners2);


    cv::Size patternsize;
    double patternMetricDistance; //3.2cm
    CalibrationPattern pattern;
    std::vector<cv::Point3f> objPoints;
};
