#pragma once
#include "Types.h"

#include <list>
#include <vector>

namespace Saiga
{
// Abstract base class for the different distribution methods
class SAIGA_VISION_API FeatureDistribution
{
public:
    FeatureDistribution(const ivec2& imageSize, int N) : imageSize(imageSize), N(N) {}
    virtual ~FeatureDistribution() {}
    /**
     * @param keypoints
     * @return The number of keypoints
     */
    virtual int operator()(std::vector<kpt_t>& keypoints) = 0;
    inline void SetN(int _N)
    {
        N = _N;
    }
    inline void SetImageSize(const ivec2& imgsz)
    {
        imageSize[0] = imgsz[0];
        imageSize[1] = imgsz[1];
    }

protected:
    // common parameters
    ivec2 imageSize;
    int N;
};

class SAIGA_VISION_API FeatureDistributionTopN : public FeatureDistribution
{
public:
    FeatureDistributionTopN(const ivec2& imageSize, int N) : FeatureDistribution(imageSize, N)
    {
    }

    int operator()(std::vector<kpt_t>& keypoints) override;
};

class SAIGA_VISION_API FeatureDistributionBucketing : public FeatureDistribution
{
public:
    FeatureDistributionBucketing(const ivec2& imageSize, int N, const ivec2& bucketSize)
            : FeatureDistribution(imageSize, N), bucketSize(bucketSize)
    {
    }

    int operator()(std::vector<kpt_t>& keypoints) override;

protected:
    ivec2 bucketSize;
};


class SAIGA_VISION_API FeatureDistributionQuadtree : public FeatureDistribution
{
public:
    FeatureDistributionQuadtree(const ivec2& imageSize, int N) : FeatureDistribution(imageSize, N)
    {
    }

    int operator()(std::vector<kpt_t>& keypoints) override;
};


class SAIGA_VISION_API FeatureDistributionANMS : public FeatureDistribution
{
public:
    enum class AccelerationStructure
    {
        KDTREE,
        RANGETREE,
        GRID
    };

    FeatureDistributionANMS(const ivec2& imageSize, int N, AccelerationStructure _ac = AccelerationStructure::GRID,
                            float _epsilon = 0.1f)
            : FeatureDistribution(imageSize, N), ac(_ac), epsilon(_epsilon)
    {
    }

    int operator()(std::vector<kpt_t>& keypoints) override;

protected:
    AccelerationStructure ac;
    float epsilon;

    int ANMSKDTree(std::vector<kpt_t>& keypoints, int high, int low);
    int ANMSRangeTree(std::vector<kpt_t>& keypoints, int high, int low);
    int ANMSGrid(std::vector<kpt_t>& keypoints, int high, int low);
};


class SAIGA_VISION_API FeatureDistributionSoftSSC : public FeatureDistribution
{
public:
    FeatureDistributionSoftSSC(const ivec2 &imageSize, int N, int _threshold = 3, float _epsilon = 0.1) :
            FeatureDistribution(imageSize, N), threshold(_threshold), epsilon(_epsilon)
    {
    }

    int operator()(std::vector<kpt_t>& keypoints) override;

protected:
    int threshold;
    float epsilon;
};

}  // namespace Saiga
