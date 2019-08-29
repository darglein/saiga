#ifndef SAIGA_ORB_SAMPLE_H
#define SAIGA_ORB_SAMPLE_H

#include <string>
#include <opencv2/core/core.hpp>
#include "saiga/vision/orb/ORBextractor.h"
#include <unistd.h>

enum Dataset
{
kitti = 0,
tum = 1,
euroc = 2
};

void SequenceMode(std::string &imgPath, int nFeatures, float scaleFactor, int nLevels, int FASTThresholdInit,
                  int FASTThresholdMin, cv::Scalar color, int thickness, int radius, bool drawAngular,
                  Dataset dataset);

void DisplayKeypoints(cv::Mat &image, std::vector<Saiga::KeyPoint> &keypoints, cv::Scalar &color,
                      int thickness = 1, int radius = 8, int drawAngular = 0, std::string windowname = "test");


void LoadImagesTUM(const std::string &strFile, std::vector<std::string> &vstrImageFilenames,
                   std::vector<double> &vTimestamps);
void LoadImagesKITTI(const std::string &strPathToSequence, std::vector<std::string> &vstrImageLeft,
                     std::vector<std::string> &vstrImageRight, std::vector<double> &vTimestamps);
void LoadImagesEUROC(const std::string &strPathLeft, const std::string &strPathRight, const std::string &strPathTimes,
                     std::vector<std::string> &vstrImageLeft, std::vector<std::string> &vstrImageRight,
                     std::vector<double> &vTimeStamps);

#endif //SAIGA_ORB_SAMPLE_H


