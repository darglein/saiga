#pragma once
#include "saiga/core/math/imath.h"
#include "saiga/core/util/Range.h"
#include "saiga/core/util/assert.h"
#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/features/Features.h"
namespace Saiga
{
template <typename T, int cell_size>
struct FeatureGridBounds2
{
    using Vec2 = Eigen::Matrix<T, 2, 1>;

    using CellId = std::pair<int, int>;

    int Rows;
    int Cols;

    // 2D axis aligned bounding box in image space
    Vec2 bmin, bmax;
    // Size (in pixels) of one cell
    Vec2 cellSize, cellSizeInv;

    std::pair<CellId, CellId> minMaxCellWithRadius(const Vec2& point, float r) const
    {
        Vec2 pmin = point.array() - r;
        Vec2 pmax = point.array() + r;
        return {cellClamped(pmin), cellClamped(pmax)};
    }

    bool inImage(const Vec2& point) const
    {
        return point(0) >= bmin(0) && point(1) >= bmin(1) && point(0) < bmax(0) && point(1) < bmax(1);
    }


    CellId LogicalCell(const Vec2& point) const
    {
        // We need the floor to down-round negative points
        Vec2 p = ((point - bmin).array() * cellSizeInv.array());
        return {iFloor(p.x()), iFloor(p.y())};
    }


    /**
     * Compute the cell id (cx,cy) for a given image point (x,y)
     */
    bool cell(const Vec2& point, CellId& cell) const
    {
        cell = LogicalCell(point);
        return inImage(point);
    }

    CellId cellClamped(const Vec2& point) const
    {
        auto cell   = LogicalCell(point);
        cell.first  = std::clamp(cell.first, 0, Cols - 1);
        cell.second = std::clamp(cell.second, 0, Rows - 1);
        return cell;
    }

    // Undistorts points at the image edges
    // to compute the bounding box
    void computeFromIntrinsicsDist(int w, int h, const Saiga::IntrinsicsPinhole<T>& intr,
                                   const Saiga::DistortionBase<T>& dist, int resolution = 20)
    {
        std::vector<Vec2> points;
        for (int i = 0; i < resolution + 1; ++i)
        {
            // Add top and bottom row
            auto alpha = float(i) / resolution * w;
            points.emplace_back(alpha, 0);
            points.emplace_back(alpha, h - 1);

            // add left and right column
            auto beta = float(i) / resolution * h;
            points.emplace_back(0, beta);
            points.emplace_back(w - 1, beta);
        }
        // undistort inplace

        std::vector<Vec2> undistortedPoints = points;
        Saiga::undistortAll(points.begin(), points.end(), undistortedPoints.begin(), intr, dist);


        // find min/max
        bmin = undistortedPoints.front();
        bmax = undistortedPoints.front();
        for (auto&& p : undistortedPoints)
        {
            // eigen coefficient-wise min/max operations
            bmin = p.array().min(bmin.array());
            bmax = p.array().max(bmax.array());
        }

        bmin -= Vec2(0.1, 0.1);
        bmax += Vec2(0.1, 0.1);

        cellSize    = Vec2(cell_size, cell_size);
        cellSizeInv = cellSize.array().inverse();


        Vec2 g_size = (bmax - bmin).array() / cellSize.array();
        Rows        = ceil(g_size.y());
        Cols        = ceil(g_size.x());
    }
};


struct FeatureGrid2
{
    using CellId = std::pair<int, int>;

    int Rows;
    int Cols;

    // Returns a permuation vector which has to be applied like this:
    //   int N = keypoints.size();
    //   std::vector<KeyPoint<float>> keypoints2(N);
    //   std::vector<Saiga::DescriptorORB> descriptors2(N);
    //
    //   for (int i = 0; i < N; ++i)
    //   {
    //       keypoints2[permutation[i]]   = keypoints[i];
    //       descriptors2[permutation[i]] = descriptors[i];
    //   }
    //   descriptors.swap(descriptors2);
    //   keypoints.swap(keypoints2);
    template <typename T, int cell_size>
    std::vector<int> create(const FeatureGridBounds2<T, cell_size>& bounds, const std::vector<KeyPoint<T>>& kps)
    {
        Rows = bounds.Rows;
        Cols = bounds.Cols;

        grid.resize(Rows, Cols);

        int N = kps.size();
        std::vector<int> permutation(N);
        std::vector<CellId> cellIds(N);

        for (int i = 0; i < Rows; ++i)
        {
            for (int j = 0; j < Cols; ++j)
            {
                grid(i, j) = {0, 0};
            }
        }

        // - Count the number of elements in each cell in grid(i,j)
        for (int i = 0; i < N; ++i)
        {
            auto& cellId = cellIds[i];
            if (bounds.cell(kps[i].point, cellId))
            {
                cell(cellId).first++;
            }
            else
            {
                cellId.first = -1;
            }
        }

        //  compute (exclusive) prefix sum over grid counts
        int totalCount = 0;
        for (int i = 0; i < Rows; ++i)
        {
            for (int j = 0; j < Cols; ++j)
            {
                auto count        = grid(i, j).first;
                grid(i, j).first  = totalCount;
                grid(i, j).second = totalCount;
                totalCount += count;
            }
        }

        // use grid.second to count the elements again and compute global target offsets for input array
        for (int i = 0; i < N; ++i)
        {
            auto& cellId = cellIds[i];
            if (cellId.first == -1)
            {
                permutation[i] = totalCount++;
                continue;
            }

            auto globalOffset = cell(cellId).second++;
            permutation[i]    = globalOffset;
        }

        SAIGA_ASSERT(totalCount == N);
        return permutation;
    }

    std::pair<int, int>& cell(CellId id)
    {
        // map (x,y) to (row,col)
        return grid(id.second, id.first);
    }



    auto cellIt(CellId id)
    {
        auto c = cell(id);
        return Range(c.first, c.second);
    }

    // Just use the eigen matrix here to get nice accessors
    // and bounds checking in debug mode.
    Eigen::Matrix<std::pair<int, int>, -1, -1, Eigen::RowMajor> grid;
};

}  // namespace Saiga
