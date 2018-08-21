#pragma once

#include "saiga/opencv/opencv.h"

namespace Saiga
{

class SAIGA_GLOBAL CalibrationPattern
{
public:
    using ImagePointType = cv::Point2f;
    using WorldPointType = cv::Point3f;

    virtual std::vector<ImagePointType> detect(cv::Mat image) = 0;

    std::vector<WorldPointType> objPoints;

    std::vector<std::vector<WorldPointType>> duplicate(int n)
    {
        return  std::vector<std::vector<WorldPointType>> (n,objPoints);
    }
};



class SAIGA_GLOBAL ChessboardPattern : public CalibrationPattern
{
public:

    ChessboardPattern(cv::Size numInnerCorners, double squareLength)
        : numInnerCorners(numInnerCorners), squareLength(squareLength)
    {
        for( int i = 0; i < numInnerCorners.height; i++ )
        {
            for( int j = 0; j < numInnerCorners.width; j++ )
            {
                objPoints.push_back(WorldPointType(j*squareLength,
                                                i*squareLength, 0));
            }
        }
    }

    std::vector<ImagePointType> detect(cv::Mat image)
    {
        std::vector<ImagePointType> corners;


        bool found = cv::findChessboardCorners( image, numInnerCorners, corners);

        if(corners.size() != objPoints.size())
            return {};

        if (found)
        {
            cv::Mat viewGray;
            cv::cvtColor(image, viewGray, CV_BGR2GRAY);
            cv::cornerSubPix( viewGray, corners, cv::Size(11,11),
                              cv::Size(-1,-1), cv::TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
        }

//        SAIGA_ASSERT(corners.size() == objPoints.size());
        if(corners.size() != objPoints.size())
            corners.clear();

        return corners;
    }
protected:
    cv::Size numInnerCorners;
    double squareLength;

};

}
