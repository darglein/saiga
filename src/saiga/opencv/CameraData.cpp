#include "CameraData.h"


namespace Saiga {


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

}
