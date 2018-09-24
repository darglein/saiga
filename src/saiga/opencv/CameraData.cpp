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

    cout << "Saved Intrinsics to " << file << endl;
}

void Intrinsics::readFromFile(string file)
{
       cv::FileStorage fs(file,FileStorage::READ);
       fs["w"] >> w;
       fs["h"] >> h;
       fs["K"] >> K;
       fs["dist"] >> dist;
}
Matx44f StereoExtrinsics::getRelativeTransform()
{
    Matx44f M = Matx44f::eye();
    for(int i = 0; i < 3; ++i)
        for(int j =0; j < 3; ++j)
            M(i,j) = R(i,j);
    for(int i = 0 ; i < 3; ++i)
        M(i,3) = t(i);
    return M;
}

void StereoExtrinsics::writeToFile(string file)
{
    cv::FileStorage fs(file,FileStorage::WRITE);

    fs << "R" << R;
    fs << "t" << t;
    fs << "F" << F;
    fs << "E" << E;

    cout << "Saved StereoExtrinsics to " << file << endl;
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
