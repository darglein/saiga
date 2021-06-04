/**
 * Copyright (c) 2021 Darius Rückert
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
    using TransformationType = SE3;
    using TangentType        = DSim3::Tangent;


    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int from = -1, to = -1;
    double weight = 1;

    DSim3 T_i_j;


    // Computes the relative pose as it is defined here
    void setRel(const SE3& T_w_i, const SE3& T_w_j)
    {
        T_i_j.se3()   = T_w_i.inverse() * T_w_j;
        T_i_j.scale() = 1.0;
    }

    void setRel(const DSim3& T_w_i, const DSim3& T_w_j) { T_i_j = T_w_i.inverse() * T_w_j; }

    // Computes the relative pose as it is defined here
    DSim3::Tangent residual(const DSim3& T_w_i, const DSim3& T_w_j) const
    {
        DSim3 T_j_i = T_w_j.inverse() * T_w_i;
        return Sophus::dsim3_logd(T_i_j * T_j_i);
    }

    SE3::Tangent residual_se3(const SE3& T_w_i, const SE3& T_w_j) const
    {
        SE3 T_j_i = T_w_j.inverse() * T_w_i;
        return Sophus::se3_logd(T_i_j.se3() * T_j_i);
    }


    SE3 GetSE3() const { return T_i_j.se3(); }

    void invert()
    {
        std::swap(from, to);
        T_i_j = T_i_j.inverse();
    }

    bool operator<(const PoseEdge& other) const { return std::tie(from, to) < std::tie(other.from, other.to); }
    explicit operator bool() const { return from >= 0 && to >= 0; }
};

struct SAIGA_VISION_API PoseVertex
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    DSim3 T_w_i;


    bool constant = false;


    SE3 Pose() const { return T_w_i.se3(); }
    DSim3 Sim3Pose() const { return T_w_i; }

    void SetPose(const SE3& v)
    {
        T_w_i.se3()   = v;
        T_w_i.scale() = 1;
    }

    void SetPose(const DSim3& v) { T_w_i = v; }
};

struct SAIGA_VISION_API PoseGraph
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    AlignedVector<PoseVertex> vertices;
    AlignedVector<PoseEdge> edges;
    bool fixScale = true;

    PoseGraph() {}
    PoseGraph(const std::string& file) { load(file); }
    PoseGraph(const Scene& scene, int minEdges = 1);

    void addNoise(double stddev);

    PoseEdge::TangentType residual6(const PoseEdge& edge);

    double density();

    double chi2();
    double rms() { return sqrt(chi2() / edges.size()); }
    void save(const std::string& file);
    void load(const std::string& file);

    // Add an edge which solidates the current vertex positions.
    void AddVertexEdge(int from, int to, double weight);

    /**
     * Ensures that i < j, and all edges are sorted by from->to.
     */
    void sortEdges();

    void transform(const SE3& T);
    void invertPoses();

    bool imgui();
};

SAIGA_VISION_API std::ostream& operator<<(std::ostream& strm, PoseGraph& pg);

}  // namespace Saiga
