#ifndef ORBEXTRACTOR_DISTRIBUTION_H
#define ORBEXTRACTOR_DISTRIBUTION_H

#include "Types.h"

#include <list>
#include <vector>
namespace SaigaORB
{
class Distribution
{
   public:
    static void DistributeKeypoints(std::vector<kpt_t>& kpts, int minX, int maxX, int minY, int maxY, int N,
                                    int softSSCThreshold = 4);

    static void DistributeKeypointsGrid(std::vector<kpt_t>& kpts, const int minX, const int maxX, const int minY,
                                        const int maxY, const int N);

   protected:
    static void DistributeKeypointsSoftSSC(std::vector<kpt_t>& kpts, int cols, int rows, int N, float epsilon,
                                           int threshold);
};



}  // namespace SaigaORB


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
    virtual int operator()(std::vector<KeyPoint<float>>& keypoints) = 0;

   protected:
    // common parameters
    ivec2 imageSize;
    int N;
};


class SAIGA_VISION_API FeatureDistributionBucketing : public FeatureDistribution
{
   public:
    FeatureDistributionBucketing(const ivec2& imageSize, int N, const ivec2& bucketSize)
        : FeatureDistribution(imageSize, N), bucketSize(bucketSize)
    {
    }

    virtual int operator()(std::vector<KeyPoint<float>>& keypoints) override
    {
        // implementation in .cpp file
        return 0;
    }

    ivec2 bucketSize;
};

}  // namespace Saiga
#endif
