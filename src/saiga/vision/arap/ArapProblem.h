/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/geometry/openMeshWrapper.h"
#include "saiga/vision/VisionIncludes.h"

namespace Saiga
{
#ifdef SAIGA_USE_OPENMESH
using ArabMesh = OpenMesh::TriMesh_ArrayKernelT<OpenMesh::DefaultTraits>;
#endif

class SAIGA_VISION_API ArapProblem
{
   public:
    struct HalfEdgeConstraint
    {
        // Vertex ids
        //        int i, j;
        std::pair<int, int> ids;
        // position_i - position_j
        Vec3 e_ij;
        double weight = 1;

        HalfEdgeConstraint() {}
        HalfEdgeConstraint(std::pair<int, int> ids, const Vec3& e_ij, double weight)
            : ids(ids), e_ij(e_ij), weight(weight)
        {
        }

        HalfEdgeConstraint flipped() { return {{ids.second, ids.first}, -e_ij, weight}; }


        bool operator<(const HalfEdgeConstraint& other) { return ids < other.ids; }

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    };

    //    AlignedVector<Vec3> vertices;
    //    AlignedVector<Quat> rotations;
    AlignedVector<SE3> vertices;
    AlignedVector<Vec3> target_positions;
    AlignedVector<int> target_indices;
    AlignedVector<HalfEdgeConstraint> constraints;

    int n;

    const double wReg = 0.01;

#ifdef SAIGA_USE_OPENMESH
    void createFromMesh(const ArabMesh& mesh);
    void saveToMesh(ArabMesh& mesh);
#endif

    double density();

    /**
     * Makes a small arapproblem for testing/debugging.
     * It contains a single triangle, with vertex 0 being moved a little upwards.
     */
    void makeTest();

    /**
     * Remove all except the first count vertices
     */
    void makeSmall(int count);
    double chi2();
};

}  // namespace Saiga
