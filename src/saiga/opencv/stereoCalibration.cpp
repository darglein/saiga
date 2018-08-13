#include "stereoCalibration.h"


using namespace cv;

using std::string;

void Intrinsics::writeToFile(string file)
{
    cv::FileStorage fs(file,FileStorage::WRITE);

    fs << "w" << w;
    fs << "h" << h;
    fs << "K" << K;
    fs << "dist" << dist;
}

void Intrinsics::readFromFile(string file)
{
       cv::FileStorage fs(file,FileStorage::READ);
       fs["w"] >> w;
       fs["h"] >> h;
       fs["K"] >> K;
       fs["dist"] >> dist;
}
void StereoExtrinsics::writeToFile(string file)
{
    cv::FileStorage fs(file,FileStorage::WRITE);

    fs << "R" << R;
    fs << "t" << t;
    fs << "F" << F;
    fs << "E" << E;
}

void StereoExtrinsics::readFromFile(string file)
{
    cv::FileStorage fs(file,FileStorage::READ);
    fs["R"] >> R;
    fs["t"] >> t;
    fs["F"] >> F;
    fs["E"] >> E;
}

StereoCalibration::StereoCalibration(cv::Size patternSize, double patternMetricDistance, CalibrationPattern pattern)
    : patternsize(patternSize), patternMetricDistance(patternMetricDistance), pattern(pattern)
{

    if(pattern == CalibrationPattern::CHESSBOARD)
    {

        for( int i = 0; i < patternsize.height; i++ )
        {
            for( int j = 0; j < patternsize.width; j++ )
            {
                objPoints.push_back(cv::Point3f(j*patternMetricDistance,
                                                i*patternMetricDistance, 0));
            }
        }
    }else{
        SAIGA_ASSERT(0);
    }
}

Intrinsics StereoCalibration::calibrateIntrinsics(std::vector<Mat> images)
{
    SAIGA_ASSERT(images.size() > 0);


    int w = images[0].cols;
    int h = images[0].rows;
    std::vector<std::vector<cv::Point2f>> corners;

    for(auto img : images)
    {
        SAIGA_ASSERT(img.cols == w && img.rows == h);
        auto pattern = findPattern(img);
        corners.push_back(pattern);
    }


    return calibrateIntrinsics(corners,w,h);

}

std::vector<cv::Point2f> StereoCalibration::findPattern(cv::Mat image)
{
    std::vector<cv::Point2f> corners;


    if(pattern == CalibrationPattern::CHESSBOARD)
    {
        bool found = cv::findChessboardCorners( image, patternsize, corners);
        if (found)
        {
            Mat viewGray;
            cvtColor(image, viewGray, CV_BGR2GRAY);
            cornerSubPix( viewGray, corners, Size(11,11),
                          Size(-1,-1), TermCriteria( CV_TERMCRIT_EPS+CV_TERMCRIT_ITER, 30, 0.1 ));
        }
    }
    return corners;
}



Intrinsics StereoCalibration::calibrateIntrinsics(
        std::vector<std::vector<Point2f>>& corners,
        int w, int h)
{
    Intrinsics intrinsics;
    if(corners.empty()) return intrinsics;
    std::vector<std::vector<Point3f>> objPointss(corners.size(),objPoints);


    intrinsics.w = w;
    intrinsics.h = h;
    Mat rvecs,tvecs;
    auto error = calibrateCamera(objPointss,corners,Size(w,h),intrinsics.K,intrinsics.dist,rvecs,tvecs);

    cout << "calibrateCamera error: " << error << endl;

    return intrinsics;
}

StereoExtrinsics StereoCalibration::calibrateStereo(
        Intrinsics intrinsics1, Intrinsics intrinsics2,
        std::vector<std::vector<Point2f> > &corners1,
        std::vector<std::vector<Point2f> > &corners2)
{
    assert(corners1.size() == corners2.size());
    StereoExtrinsics extr;
    if(corners1.empty()) return extr;
    std::vector<std::vector<Point3f>> objPointss(corners1.size(),objPoints);


    auto error = cv::stereoCalibrate(objPointss,corners1,corners2,intrinsics1.K,intrinsics1.dist,intrinsics2.K,intrinsics2.dist,Size(intrinsics1.w,intrinsics1.h),extr.R,extr.t,extr.E,extr.F,CALIB_FIX_INTRINSIC);

    cout << "stereoCalibrate error: " << error << endl;

    return extr;
}



