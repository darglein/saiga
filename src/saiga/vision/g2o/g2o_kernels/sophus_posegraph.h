/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 *
 * This file is a partial copy of LSD-SLAM by Jakob Engel.
 * LSD-SLAM was resleased under the GNU General Public license.
 * For more information see <http://vision.in.tum.de/lsdslam>
 *
 */

#pragma once

#include "saiga/util/assert.h"
#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/pgo/PGOConfig.h"

#include "g2o/core/base_binary_edge.h"
#include "g2o/core/base_unary_edge.h"
#include "g2o/core/base_vertex.h"

namespace Saiga
{
class SAIGA_GLOBAL VertexSim3 : public g2o::BaseVertex<6, SE3>
{
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexSim3() { _marginalized = false; }
    virtual bool read(std::istream&) { return true; }
    virtual bool write(std::ostream&) const { return true; }

    virtual void setToOriginImpl() { _estimate = SE3(); }

    virtual void oplusImpl(const double* update_)
    {
        Eigen::Map<Eigen::Matrix<double, 6, 1> > update(const_cast<double*>(update_));
        setEstimate(SE3::exp(update) * estimate());
    }
};

/**
 * \brief 7D edge between two Vertex7
 */
class SAIGA_GLOBAL EdgeSim3 : public g2o::BaseBinaryEdge<6, SE3, VertexSim3, VertexSim3>
{
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeSim3() {}

    virtual bool read(std::istream&) { return true; }
    virtual bool write(std::ostream&) const { return false; }

    void computeError()
    {
        const VertexSim3* _from = static_cast<const VertexSim3*>(_vertices[0]);
        const VertexSim3* _to   = static_cast<const VertexSim3*>(_vertices[1]);
#ifdef LSD_REL
        SE3 error_ = _from->estimate().inverse() * _to->estimate() * _inverseMeasurement;
#else
        SE3 error_ = _inverseMeasurement * _to->estimate() * _from->estimate().inverse();
//        SE3 error_       = _to->estimate() * _from->estimate().inverse() * _inverseMeasurement;
#endif
        _error = error_.log();
    }

    void linearizeOplus()
    {
#ifdef LSD_REL
        const VertexSim3* _from = static_cast<const VertexSim3*>(_vertices[0]);
        _jacobianOplusXj        = _from->estimate().inverse().Adj();
        _jacobianOplusXi        = -_jacobianOplusXj;
#else
        const VertexSim3* _to = static_cast<const VertexSim3*>(_vertices[1]);
        SE3 error             = _inverseMeasurement * _to->estimate() * _from->estimate().inverse();
        _jacobianOplusXj      = error.Adj();
        _jacobianOplusXi      = -_jacobianOplusXj;

#endif
    }


    virtual void setMeasurement(const SE3& m)
    {
        _measurement        = m;
        _inverseMeasurement = m.inverse();
    }

    virtual bool setMeasurementFromState()
    {
        const VertexSim3* from = static_cast<const VertexSim3*>(_vertices[0]);
        const VertexSim3* to   = static_cast<const VertexSim3*>(_vertices[1]);
#ifdef LSD_REL
        SE3 delta = from->estimate().inverse() * to->estimate();
#else
        SE3 delta             = to->estimate() * from->estimate().inverse();
#endif
        setMeasurement(delta);
        return true;
    }


   protected:
    SE3 _inverseMeasurement;
};

}  // namespace Saiga
