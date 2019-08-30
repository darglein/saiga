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
#endif
