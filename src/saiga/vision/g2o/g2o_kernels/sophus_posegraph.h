/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 *
 * This file is a partial copy of LSD-SLAM by Jakob Engel.
 * LSD-SLAM was resleased under the GNU General Public license.
 * For more information see <http://vision.in.tum.de/lsdslam>
 *
 */

#pragma once

#include "saiga/core/util/assert.h"
#include "saiga/vision/VisionTypes.h"
#include "saiga/vision/pgo/PGOConfig.h"

#include "g2o/core/base_binary_edge.h"
#include "g2o/core/base_unary_edge.h"
#include "g2o/core/base_vertex.h"

namespace Saiga
{
using G2OSim3VertexType = SE3;

class SAIGA_VISION_API VertexSim3 : public g2o::BaseVertex<G2OSim3VertexType::DoF, G2OSim3VertexType>
{
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    VertexSim3() { _marginalized = false; }
    virtual bool read(std::istream&) { return true; }
    virtual bool write(std::ostream&) const { return true; }

    virtual void setToOriginImpl() { _estimate = G2OSim3VertexType(); }

    virtual void oplusImpl(const double* update_)
    {
        Eigen::Map<Eigen::Matrix<double, G2OSim3VertexType::DoF, 1> > update(const_cast<double*>(update_));
#ifdef PGO_SIM3
        if (fixScale) update[6] = 0;
#endif
        //#ifdef LSD_REL

        G2OSim3VertexType s = G2OSim3VertexType::exp(update);
        setEstimate(s * estimate());
        //#else
        //        //        setEstimate(estimate() * SE3::exp(update));
        //        setEstimate(SE3::exp(update) * estimate());
        //#endif
    }
    bool fixScale = true;
};

/**
 * \brief 7D edge between two Vertex7
 */
template <bool _LSD_REL = true>
class SAIGA_VISION_API EdgeSim3
    : public g2o::BaseBinaryEdge<G2OSim3VertexType::DoF, G2OSim3VertexType, VertexSim3, VertexSim3>
{
   public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    EdgeSim3() {}

    virtual bool read(std::istream&) { return true; }
    virtual bool write(std::ostream&) const { return false; }

    void computeError()
    {
        auto from = static_cast<const VertexSim3*>(_vertices[0])->estimate();
        auto to   = static_cast<const VertexSim3*>(_vertices[1])->estimate();
        if (_LSD_REL)
        {
            G2OSim3VertexType error_ = from.inverse() * to * _inverseMeasurement;
            _error                   = error_.log();
        }
        else
        {
            G2OSim3VertexType error_ = _measurement * from * to.inverse();
            _error                   = error_.log();
        }
    }

    void linearizeOplus()
    {
        if (_LSD_REL)
        {
            auto from        = static_cast<const VertexSim3*>(_vertices[0])->estimate();
            _jacobianOplusXj = from.inverse().Adj();
            _jacobianOplusXi = -_jacobianOplusXj;
        }
        else
        {
            g2o::BaseBinaryEdge<G2OSim3VertexType::DoF, G2OSim3VertexType, VertexSim3, VertexSim3>::linearizeOplus();
        }
    }


    virtual void setMeasurement(const G2OSim3VertexType& m)
    {
        _measurement        = m;
        _inverseMeasurement = m.inverse();
    }

    virtual bool setMeasurementFromState()
    {
        SAIGA_EXIT_ERROR("");
        const VertexSim3* from  = static_cast<const VertexSim3*>(_vertices[0]);
        const VertexSim3* to    = static_cast<const VertexSim3*>(_vertices[1]);
        G2OSim3VertexType delta = from->estimate().inverse() * to->estimate();
        setMeasurement(delta);
        return true;
    }


   protected:
    G2OSim3VertexType _inverseMeasurement;
};

}  // namespace Saiga
