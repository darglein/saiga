/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once

#include "saiga/config.h"
#include "saiga/core/image/image.h"
#include "saiga/core/util/statistics.h"
#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/pgo/PGOConfig.h"
#include "saiga/vision/scene/Scene.h"

#include <vector>


namespace Saiga
{
struct SAIGA_VISION_API PoseEdge
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int from = -1, to = -1;
    double weight = 1;
    SE3 meassurement;

    // Computes the relative pose as it is defined here
    void setRel(const SE3& from, const SE3& to)
    {
#ifdef LSD_REL
        meassurement = from.inverse() * to;
#else
        meassurement = to * from.inverse();
#endif
    }

    // Computes the relative pose as it is defined here
    Vec6 residual(const SE3& from, const SE3& to)
    {
#ifdef LSD_REL
        auto error_ = from.inverse() * to * meassurement.inverse();
#else
        auto error_  = meassurement * from * to.inverse();
#endif
        return error_.log() * weight;
    }

    void invert()
    {
        std::swap(from, to);
        meassurement = meassurement.inverse();
    }

    bool operator<(const PoseEdge& other) { return std::tie(from, to) < std::tie(other.from, other.to); }
    explicit operator bool() const { return from >= 0 && to >= 0; }
};

struct SAIGA_VISION_API PoseVertex
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SE3 se3;
    bool constant = false;
};

struct SAIGA_VISION_API PoseGraph
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    AlignedVector<PoseVertex> poses;
    AlignedVector<PoseEdge> edges;

    PoseGraph() {}
    PoseGraph(const std::string& file) { load(file); }
    PoseGraph(const Scene& scene, int minEdges = 1);

    void addNoise(double stddev);

    Vec6 residual6(const PoseEdge& edge);

    double density();

    double chi2();
    double rms() { return sqrt(chi2() / edges.size()); }
    void save(const std::string& file);
    void load(const std::string& file);

    /**
     * Ensures that i < j, and all edges are sorted by from->to.
     */
    void sortEdges();

    bool imgui();
};

SAIGA_VISION_API std::ostream& operator<<(std::ostream& strm, PoseGraph& pg);

}  // namespace Saiga
