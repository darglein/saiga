#ifndef SAIGA_ORB_FAST_H
#define SAIGA_ORB_FAST_H

#include "ORBconstants.h"
#include "Types.h"

namespace Saiga
{
// const int CIRCLE_SIZE = 16;

const int CIRCLE_OFFSETS[16][2] = {{0, 3},  {1, 3},   {2, 2},   {3, 1},   {3, 0},  {3, -1}, {2, -2}, {1, -3},
                                   {0, -3}, {-1, -3}, {-2, -2}, {-3, -1}, {-3, 0}, {-3, 1}, {-2, 2}, {-1, 3}};

const int PIXELS_TO_CHECK[16] = {0, 8, 2, 10, 4, 12, 6, 14, 1, 9, 3, 11, 5, 13, 7, 15};

class FASTdetector
{
   public:
    FASTdetector(int _iniThreshold, int _minThreshold, int _nlevels);

    ~FASTdetector() = default;

    void SetStepVector(std::vector<int>& _steps);

    void SetFASTThresholds(int ini, int min);

    void FAST(img_t img, std::vector<kpt_t>& keypoints, int threshold, int lvl);

    void inline SetLevels(int nlvls) { pixelOffset.resize(nlvls * CIRCLE_SIZE); }

   protected:
    int iniThreshold;
    int minThreshold;

    int nlevels;

    int continuousPixelsRequired;
    int onePointFiveCircles;

    std::vector<int> pixelOffset;
    std::vector<int> steps;

    uchar threshold_tab_init[512];
    uchar threshold_tab_min[512];


    template <typename scoretype>
    void FAST_t(img_t& img, std::vector<kpt_t>& keypoints, int threshold, int lvl);

    float CornerScore(const uchar* pointer, const int offset[], int threshold);
};

}  // namespace SaigaORB
#endif  // SAIGA_ORB_FAST_H
