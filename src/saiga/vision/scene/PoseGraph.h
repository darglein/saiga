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
    using TransformationType = SE3;
    using TangentType        = SE3::Tangent;


    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    int from = -1, to = -1;
    double weight = 1;

    SE3 T_i_j;
    double scale = 1.0;

    // Computes the relative pose as it is defined here
    void setRel(const SE3& T_w_i, const SE3& T_w_j)
    {
        T_i_j = T_w_i.inverse() * T_w_j;
        scale = 1.0;
    }

    void setRel(const Sim3& T_w_i, const Sim3& T_w_j)
    {
        auto ss_T_w_i = se3Scale(T_w_i);
        auto ss_T_w_j = se3Scale(T_w_i);
        setRel(ss_T_w_i.first, ss_T_w_j.first);

        scale = (1.0 / ss_T_w_j.second) * ss_T_w_i.second;
    }

    // Computes the relative pose as it is defined here
    TangentType residual(const TransformationType& T_w_i, const TransformationType& T_w_j) const
    {
        TransformationType T_j_i = T_w_j.inverse() * T_w_i;
        return Sophus::se3_logd(T_i_j * T_j_i);
    }

    TransformationType meassurement() const { return T_i_j; }

    void invert()
    {
        std::swap(from, to);
        T_i_j = T_i_j.inverse();
    }

    bool operator<(const PoseEdge& other) { return std::tie(from, to) < std::tie(other.from, other.to); }
    explicit operator bool() const { return from >= 0 && to >= 0; }
};

struct SAIGA_VISION_API PoseVertex
{
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    SE3 T_w_i;
    double scale = 1.0;

    bool constant = false;


    SE3 Pose() const { return T_w_i; }
    Sim3 Sim3Pose() const { return sim3(T_w_i, scale); }

    void SetPose(const SE3& v)
    {
        T_w_i = v;
        scale = 1;
    }

    void SetPose(const Sim3& v)
    {
        auto ss = se3Scale(v);
        T_w_i   = ss.first;
        scale   = ss.second;
    }
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
