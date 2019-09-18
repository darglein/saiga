#include "FeatureDistribution.h"

#include <algorithm>
#include <vector>

#ifdef ORB_USE_OPENCV
#    include <opencv2/core/core.hpp>
#endif

namespace Saiga
{
static void RetainBestN(std::vector<kpt_t>& kpts, int N)
{
    if (kpts.size() <= N) return;
    std::nth_element(kpts.begin(), kpts.begin() + N, kpts.end(),
                     [](const kpt_t& k1, const kpt_t& k2) { return k1.response > k2.response; });
    kpts.resize(N);
}


void Distribution::DistributeKeypoints(std::vector<kpt_t>& kpts, const int minX, const int maxX, const int minY,
                                       const int maxY, const int N, int softSSCThreshold)
{
    if (kpts.size() <= N) return;
    if (N == 0)
    {
        kpts = std::vector<kpt_t>(0);
        return;
    }
    const float epsilon = 0.1;


    std::sort(kpts.begin(), kpts.end(), [](const kpt_t& k1, const kpt_t& k2) { return k1.response > k2.response; });

    int cols = maxX - minX;
    int rows = maxY - minY;
    //    DistributeKeypointsSoftSSC(kpts, cols, rows, N, epsilon, softSSCThreshold);
    DistributeKeypointsGrid(kpts, minX, maxX, minY, maxY, N);
}

void Distribution::DistributeKeypointsGrid(std::vector<kpt_t>& kpts, const int minX, const int maxX, const int minY,
                                           const int maxY, const int N)
{
    // std::sort(kpts.begin(), kpts.end(), [](const cv::KeyPoint &a, const cv::KeyPoint &b){return (a.pt.x < b.pt.x ||
    //        (a.pt.x == b.pt.x && a.pt.y < b.pt.y));});

    const float width  = maxX - minX;
    const float height = maxY - minY;

    int cellSize = (int)std::min(80.f, std::min(width, height));

    const int npatchesInX = width / cellSize;
    const int npatchesInY = height / cellSize;
    const int patchWidth  = ceil(width / npatchesInX);
    const int patchHeight = ceil(height / npatchesInY);

    int nCells = npatchesInX * npatchesInY;
    std::vector<std::vector<kpt_t>> cellkpts(nCells);
    int nPerCell = (float)N / nCells;


    for (auto& kpt : kpts)
    {
        int idx = (int)(kpt.point(1) / patchHeight) * npatchesInX + (int)(kpt.point(0) / patchWidth);
        if (idx >= nCells) idx = nCells - 1;
        cellkpts[idx].emplace_back(kpt);
    }

    kpts.clear();
    kpts.reserve(N * 2);

    for (auto& kptVec : cellkpts)
    {
        RetainBestN(kptVec, nPerCell);
        kpts.insert(kpts.end(), kptVec.begin(), kptVec.end());
    }
}



void Distribution::DistributeKeypointsSoftSSC(std::vector<kpt_t>& kpts, const int cols, const int rows, int N,
                                              float epsilon, int threshold)
{
    int numerator1         = rows + cols + 2 * N;
    long long discriminant = (long long)4 * cols + (long long)4 * N + (long long)4 * rows * N + (long long)rows * rows +
                             (long long)cols * cols - (long long)2 * cols * rows + (long long)4 * cols * rows * N;

    double denominator = 2 * (N - 1);

    double sol1 = (numerator1 - sqrt(discriminant)) / denominator;
    double sol2 = (numerator1 + sqrt(discriminant)) / denominator;

    int high = (sol1 > sol2) ? sol1 : sol2;
    int low  = std::floor(sqrt((double)kpts.size() / N));

    bool done = false;
    int kMin = round(N - N * epsilon), kMax = round(N + N * epsilon);
    std::vector<int> resultIndices;
    int row, col, width, prevwidth = -1;

    std::vector<int> tempResult;
    tempResult.reserve(kpts.size());

    while (!done)
    {
        width = low + (high - low) / 2;
        if (width == prevwidth || low > high)
        {
            resultIndices = tempResult;
            break;
        }
        tempResult.clear();
        double c     = (double)width / 2.0;
        int cellCols = std::floor(cols / c);
        int cellRows = std::floor(rows / c);
        std::vector<std::vector<int>> covered(cellRows + 1, std::vector<int>(cellCols + 1, -1));

        for (int i = 0; i < (int)kpts.size(); ++i)
        {
            row = (int)((kpts[i].point.y()) / c);
            col = (int)((kpts[i].point.x()) / c);


            int score = (int)round((double)kpts[i].response);

            if (covered[row][col] < score + threshold)
            {
                tempResult.emplace_back(i);
                int rowMin = row - (int)(width / c) >= 0 ? (row - (int)(width / c)) : 0;
                int rowMax = row + (int)(width / c) <= cellRows ? (row + (int)(width / c)) : cellRows;
                int colMin = col - (int)(width / c) >= 0 ? (col - (int)(width / c)) : 0;
                int colMax = col + (int)(width / c) <= cellCols ? (col + (int)(width / c)) : cellCols;

                for (int dy = rowMin; dy <= rowMax; ++dy)
                {
                    for (int dx = colMin; dx <= colMax; ++dx)
                    {
                        if (covered[dy][dx] < score) covered[dy][dx] = score;
                    }
                }
            }
        }
        if (tempResult.size() >= kMin && tempResult.size() <= kMax)
        {
            resultIndices = tempResult;
            done          = true;
        }
        else if (tempResult.size() < kMin)
            high = width - 1;
        else
            low = width + 1;

        prevwidth = width;
    }

    std::vector<kpt_t> reskpts;
    for (int i = 0; i < resultIndices.size(); ++i)
    {
        reskpts.emplace_back(kpts[resultIndices[i]]);
    }
    kpts = reskpts;
}
}  // namespace Saiga
