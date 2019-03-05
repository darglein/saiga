/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#pragma once
#include "saiga/core/geometry/openMeshWrapper.h"
#include "saiga/vision/VisionIncludes.h"

namespace Saiga
{
using ArabMesh = OpenMesh::TriMesh_ArrayKernelT<OpenMesh::DefaultTraits>;


class ArapProblem
{
   public:
    struct HalfEdgeConstraint
    {
        // Vertex ids
        int i, j;
        // position_i - position_j
        Vec3 e_ij;
        double weight = 1;

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


    void createFromMesh(const ArabMesh& mesh);
    void saveToMesh(ArabMesh& mesh);
};

}  // namespace Saiga
