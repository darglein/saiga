#ifndef SAIGA_ORB_ORBEXTRACTOR_H
#define SAIGA_ORB_ORBEXTRACTOR_H

#include "FAST.h"
#include "FeatureDistribution.h"

#include <vector>
#ifdef ORB_USE_OPENCV
#    include <opencv2/imgproc/imgproc.hpp>
#endif

#ifdef ORB_FEATURE_FILEINTERFACE_ENABLED
#    include "include/FeatureFileInterface.h"
#endif



namespace Saiga
{
struct Point2i
{
    int x;
    int y;

    Point2i() : x(0), y(0) {}

    Point2i(int _x, int _y) : x(_x), y(_y) {}

    bool inline operator==(const Point2i& other) const { return x == other.x && y == other.y; }

    template <typename T>
    inline friend Point2i operator*(const T s, const Point2i& pt)
    {
        return Point(pt.x * s, pt.y * s);
    }

    template <typename T>
    inline friend void operator*=(Point2i& pt, const T s)
    {
        pt.x *= s;
        pt.y *= s;
    }

    friend std::ostream& operator<<(std::ostream& os, const Point2i& pt)
    {
        os << "[" << pt.x << "," << pt.y << "]";
        return os;
    }
};

class SAIGA_VISION_API ORBextractor
{
   public:
    ORBextractor(int nfeatures, float scaleFactor, int nlevels, int iniThFAST, int minThFAST);

    ~ORBextractor() = default;

#ifdef ORB_USE_OPENCV
    void operator()(cv::InputArray image, cv::InputArray mask, std::vector<kpt_t>& keypoints,
                    cv::OutputArray descriptors, FeatureDistribution& distribution);
#endif


    void operator()(Saiga::ImageView<uchar> inputImage, std::vector<kpt_t>& resultKeypoints,
                    Saiga::TemplatedImage<uchar>& outputDescriptors, FeatureDistribution& distribution,
                    bool distributePerLevel = true);

    int inline GetLevels() { return nlevels; }

    float inline GetScaleFactor() { return scaleFactor; }

    std::vector<float> inline GetScaleFactors() { return scaleFactorVec; }

    std::vector<float> inline GetInverseScaleFactors() { return invScaleFactorVec; }

    std::vector<float> inline GetScaleSigmaSquares() { return levelSigma2Vec; }

    std::vector<float> inline GetInverseScaleSigmaSquares() { return invLevelSigma2Vec; }

    void inline SetLevelToDisplay(int lvl) { levelToDisplay = std::min(lvl, nlevels - 1); }

    void inline SetSoftSSCThreshold(float th) { softSSCThreshold = th; }

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
    void inline SetFeatureSaving(bool s) { saveFeatures = s; }
    void inline SetLoadPath(std::string& path) { loadPath = path; }
    void inline EnablePrecomputedFeatures(bool b) { usePrecomputedFeatures = b; }
    inline FeatureFileInterface* GetFileInterface() { return &fileInterface; }
#endif

    void SetSteps();

   protected:
    static float IntensityCentroidAngle(const uchar* pointer, int step);


    void ComputeAngles(std::vector<std::vector<kpt_t>>& allkpts);

    void ComputeDescriptors(std::vector<std::vector<kpt_t>>& allkpts, img_t& descriptors);


    void DivideAndFAST(std::vector<std::vector<kpt_t>>& allkpts, FeatureDistribution& distribution, int cellSize = 30, bool distributePerLevel = true);
#ifdef ORB_USE_OPENCV
    void ComputeScalePyramid(img_t& image, std::vector<cv::Mat>& tmpPyramid);
#endif

    std::vector<Point2i> pattern;

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

    int softSSCThreshold = 1;

    Point2i prevDims;

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
}  // namespace Saiga

#endif  // SAIGA_ORB_ORBEXTRACTOR_H
