#ifndef SAIGA_ORB_ORBEXTRACTOR_H
#define SAIGA_ORB_ORBEXTRACTOR_H

#include <vector>
#include "Distribution.h"
#include "FAST.h"
#include <opencv2/imgproc/imgproc.hpp>

#ifdef ORB_FEATURE_FILEINTERFACE_ENABLED
#include "include/FeatureFileInterface.h"
#endif



namespace SaigaORB
{

class ORBextractor
{
public:

    ORBextractor(int nfeatures, float scaleFactor, int nlevels,
                 int iniThFAST, int minThFAST);

    ~ORBextractor() = default;


    void operator()( cv::InputArray image, cv::InputArray mask,
                     std::vector<Saiga::KeyPoint>& keypoints,
                     cv::OutputArray descriptors);


    void operator()(Saiga::ImageView<uchar> &inputImage, std::vector<Saiga::KeyPoint> &resultKeypoints,
                    Saiga::ImageView<uchar> &outputDescriptors, bool distributePerLevel);

    int inline GetLevels(){
        return nlevels;}

    float inline GetScaleFactor(){
        return scaleFactor;}

    std::vector<float> inline GetScaleFactors(){
        return scaleFactorVec;
    }

    std::vector<float> inline GetInverseScaleFactors(){
        return invScaleFactorVec;
    }

    std::vector<float> inline GetScaleSigmaSquares(){
        return levelSigma2Vec;
    }

    std::vector<float> inline GetInverseScaleSigmaSquares(){
        return invLevelSigma2Vec;
    }

    void inline SetLevelToDisplay(int lvl)
    {
        levelToDisplay = std::min(lvl, nlevels-1);
    }

    void inline SetSoftSSCThreshold(float th)
    {
        softSSCThreshold = th;
    }

    void SetnFeatures(int n);

    void SetFASTThresholds(int ini, int min);

    void SetnLevels(int n);

    void SetScaleFactor(float s);

#ifdef _FEATURE_FILEINTERFACE_ENABLED
    void SetFeatureSavePath(std::string& path)
    {
        path += std::to_string(nfeatures) + "f_" + std::to_string(scaleFactor) + "s_" +
                std::to_string(kptDistribution) + "d/";
        fileInterface.SetPath(path);
    }
    void inline SetFeatureSaving(bool s)
    {
        saveFeatures = s;
    }
    void inline SetLoadPath(std::string &path)
    {
        loadPath = path;
    }
    void inline EnablePrecomputedFeatures(bool b)
    {
        usePrecomputedFeatures = b;
    }
    inline FeatureFileInterface* GetFileInterface()
    {
        return &fileInterface;
    }
#endif

    void SetSteps();

protected:

    static float IntensityCentroidAngle(const uchar* pointer, int step);


    void ComputeAngles(std::vector<std::vector<Saiga::KeyPoint>> &allkpts);

    void ComputeDescriptors(std::vector<std::vector<Saiga::KeyPoint>> &allkpts, img_t &descriptors);


    void DivideAndFAST(std::vector<std::vector<Saiga::KeyPoint>>& allkpts, int cellSize = 30,
            bool distributePerLevel = true);

    void ComputeScalePyramid(img_t& image, std::vector<cv::Mat>& tmpPyramid);

    std::vector<Saiga::Point> pattern;
public:
    std::vector<img_t> imagePyramid;
protected:
    int nfeatures;
    double scaleFactor;
    int nlevels;
    int iniThFAST;
    int minThFAST;
    bool stepsChanged;

    int levelToDisplay;

    int softSSCThreshold;

    Saiga::Point prevDims;

    std::vector<int> pixelOffset;

    std::vector<int> nfeaturesPerLevelVec;
    std::vector<int> featuresPerLevelActual;


    std::vector<float> scaleFactorVec;
    std::vector<float> invScaleFactorVec;
    std::vector<float> levelSigma2Vec;
    std::vector<float> invLevelSigma2Vec;

    FASTdetector fast;

#ifdef _FEATURE_FILEINTERFACE_ENABLED
    FeatureFileInterface fileInterface;
    bool saveFeatures;
    bool usePrecomputedFeatures;
    std::string loadPath;
#endif
};
}

#endif //SAIGA_ORB_ORBEXTRACTOR_H