// g2o - General Graph Optimization
// Copyright (C) 2011 H. Strasdat
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// * Redistributions of source code must retain the above copyright notice,
//   this list of conditions and the following disclaimer.
// * Redistributions in binary form must reproduce the above copyright
//   notice, this list of conditions and the following disclaimer in the
//   documentation and/or other materials provided with the distribution.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
// IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
// TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
// PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
// TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#pragma once

#include "saiga/core/util/assert.h"
#include "saiga/vision/VisionTypes.h"

#include "g2o/core/base_binary_edge.h"
#include "g2o/core/base_unary_edge.h"
#include "g2o/core/base_vertex.h"

namespace g2o
{
using Saiga::SE3;
using Saiga::Vec2;
using Saiga::Vec3;

/**
 * \brief SE3 Vertex parameterized internally with a transformation matrix
 and externally with its exponential map
 */
class VertexSE3 : public g2o::BaseVertex<6, SE3>
{
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    VertexSE3();
    bool read(std::istream& is) { return false; }
    bool write(std::ostream& os) const { return false; }
    virtual void setToOriginImpl() { _estimate = SE3(); }

    virtual void oplusImpl(const number_t* update_)
    {
        Eigen::Map<const SE3::Tangent> update(update_);

        SE3::Tangent update2;

        update2(0) = update(3);
        update2(1) = update(4);
        update2(2) = update(5);
        update2(3) = update(0);
        update2(4) = update(1);
        update2(5) = update(2);

        update2 = update;

        //    std::cout << "update " << update2 << std::endl;
        //    std::cout << "update se3 " << update2.transpose() << std::endl;
        SAIGA_ASSERT(update2.allFinite());
        setEstimate(SE3::exp(update2) * estimate());
    }
};

/**
 * \brief Point vertex, XYZ
 */
class VertexPoint : public g2o::BaseVertex<3, Vec3>
{
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    VertexPoint();
    bool read(std::istream& is) { return false; }
    bool write(std::ostream& os) const { return false; }
    virtual void setToOriginImpl() { _estimate.fill(0); }

    virtual void oplusImpl(const number_t* update)
    {
        Eigen::Map<const Vec3> v(update);
        _estimate += v;
    }
};

// Projection using focal_length in x and y directions
class EdgeSE3PointProject : public g2o::BaseBinaryEdge<2, Vec2, VertexPoint, VertexSE3>
{
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3PointProject();

    bool read(std::istream& is) { return false; }
    bool write(std::ostream& os) const { return false; }

    void computeError()
    {
        SE3 se3 = static_cast<const VertexSE3*>(_vertices[1])->estimate();
        Vec3 wp = static_cast<const VertexPoint*>(_vertices[0])->estimate();
        Vec2 observed(_measurement(0), _measurement(1));

        Vec3 cp   = se3 * wp;
        Vec2 ip   = intr.project(cp);
        Vec2 diff = observed - ip;

        _error(0) = diff(0) * weight;
        _error(1) = diff(1) * weight;
    }

    virtual void linearizeOplus();


    Saiga::IntrinsicsPinholed intr;
    double weight;
};


// Projection using focal_length in x and y directions stereo
class EdgeSE3PointProjectDepth : public g2o::BaseBinaryEdge<3, Vec3, VertexPoint, VertexSE3>
{
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    EdgeSE3PointProjectDepth();

    bool read(std::istream& is) { return false; }
    bool write(std::ostream& os) const { return false; }

    void computeError()
    {
        SE3 se3 = static_cast<const VertexSE3*>(_vertices[1])->estimate();
        Vec3 wp = static_cast<const VertexPoint*>(_vertices[0])->estimate();
        Vec2 observed(_measurement(0), _measurement(1));
        double stereoPoint = _measurement(2);

        Vec3 cp        = se3 * wp;
        Vec2 ip        = intr.project(cp);
        Vec2 diff      = observed - ip;
        auto disparity = ip(0) - bf / cp(2);
        double diff2   = stereoPoint - disparity;


        _error(0) = diff(0) * weights(0);
        _error(1) = diff(1) * weights(0);
        _error(2) = diff2 * weights(1);
    }


    virtual void linearizeOplus();


    Saiga::IntrinsicsPinholed intr;
    double bf;
    Vec2 weights;
};


}  // namespace g2o
