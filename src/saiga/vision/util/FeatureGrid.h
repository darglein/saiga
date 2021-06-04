/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/math/imath.h"
#include "saiga/core/util/Range.h"
#include "saiga/core/util/assert.h"
#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/features/Features.h"
namespace Saiga
{
template <typename T, int _Rows, int _Cols>
struct FeatureGridBounds
{
    using Vec2 = Eigen::Matrix<T, 2, 1>;

    using CellId = std::pair<int, int>;

    static constexpr int Rows = _Rows;
    static constexpr int Cols = _Cols;

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
        return (point(0) >= bmin(0)) & (point(1) >= bmin(1)) & (point(0) < bmax(0)) & (point(1) < bmax(1));
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
                                   const Saiga::DistortionBase<T>& dist, int resolution = 10)
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

        cellSize    = (bmax - bmin).array() / Vec2(Cols, Rows).array();
        cellSizeInv = cellSize.array().inverse();

        // Debug output
        std::cout << "Feature Undistorted Bounding box:" << std::endl;
        std::cout << "[" << bmin.transpose() << "] [" << bmax.transpose() << "]" << std::endl;
        std::cout << Cols << "x" << Rows << std::endl;
        std::cout << "Cell Size: " << cellSize.transpose() << std::endl;
    }
};

template <int _Rows, int _Cols>
struct FeatureGrid
{
    using CellId = std::pair<int, int>;

    static constexpr int Rows = _Rows;
    static constexpr int Cols = _Cols;

    template <typename T>
    std::vector<int> create(const FeatureGridBounds<T, _Rows, _Cols>& bounds, const std::vector<KeyPoint<T>>& kps)
    {
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

        // - Compute for each kp the cell id and store it int permutation
        // - Count the number of elements in each cell in grid(i,j).first
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

#if 0
        {
            // Some debug checks + prints

            // Permute kp array
            std::vector<KeyPoint<T>> cpy(N);
            for (int i = 0; i < N; ++i)
            {
                cpy[permutation[i]] = kps[i];
            }

            // Go for each cell over all keypoints and check if they project to
            // this cell
            for (int i = 0; i < Rows; ++i)
            {
                for (int j = 0; j < Cols; ++j)
                {
                    for (int c : cellIt({j, i}))
                    {
                        CellId cid2;
                        bounds.cell(cpy[c].point, cid2);
                        SAIGA_ASSERT(cid2.first == j && cid2.second == i);
                    }
                }
            }
            std::cout << "Grid debug check OK!" << std::endl;
        }
#endif


        return permutation;
    }

    auto& cell(CellId id)
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
    Eigen::Matrix<std::pair<int, int>, Rows, Cols, Eigen::RowMajor> grid;
};

}  // namespace Saiga
