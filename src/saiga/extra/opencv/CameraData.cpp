#include "CameraData.h"


namespace Saiga
{
using namespace cv;

using std::string;

void Intrinsics::writeToFile(string file)
{
    cv::FileStorage fs(file, FileStorage::WRITE);

    fs << "w" << w;
    fs << "h" << h;
    fs << "K" << K;
    fs << "dist" << dist;

    std::cout << "Saved Intrinsics to " << file << std::endl;
}

bool Intrinsics::readFromFile(string file)
{
    cv::FileStorage fs(file, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["w"] >> w;
    fs["h"] >> h;
    fs["K"] >> K;
    fs["dist"] >> dist;

    std::cout << "Loaded Intrinsics from " << file << std::endl;
    return true;
}
Matx44f StereoExtrinsics::getRelativeTransform()
{
    Matx44f M = Matx44f::eye();
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) M(i, j) = R(i, j);
    for (int i = 0; i < 3; ++i) M(i, 3) = t(i);
    return M;
}

void StereoExtrinsics::writeToFile(string file)
{
    cv::FileStorage fs(file, FileStorage::WRITE);

    fs << "R" << R;
    fs << "t" << t;
    fs << "F" << F;
    fs << "E" << E;

    std::cout << "Saved StereoExtrinsics to " << file << std::endl;
}

bool StereoExtrinsics::readFromFile(string file)
{
    cv::FileStorage fs(file, FileStorage::READ);
    if(!fs.isOpened())
        return false;
    fs["R"] >> R;
    fs["t"] >> t;
    fs["F"] >> F;
    fs["E"] >> E;

    std::cout << "Loaded StereoExtrinsics from " << file << std::endl;
    return true;
}

}  // namespace Saiga
