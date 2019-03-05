/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ArapProblem.h"

namespace Saiga
{
void ArapProblem::createFromMesh(const ArabMesh& mesh)
{
    vertices.clear();
    target_indices.clear();
    target_positions.clear();

    n = mesh.n_vertices();

    for (int i = 0; i < n; ++i)
    {
        auto v  = mesh.vertex_handle(i);
        auto p  = mesh.point(v);
        Vec3 pd = {p[0], p[1], p[2]};

        SE3 se3;
        se3.translation() = pd;
        vertices.push_back(se3);
    }


    for (int i = 0; i < n; ++i)
    {
        auto v_i = mesh.vertex_handle(i);

        auto& p = vertices[i];

        for (auto v_j : mesh.vv_range(v_i))
        {
            int j   = v_j.idx();
            auto& q = vertices[j];

            HalfEdgeConstraint hec;
            hec.i = i;
            hec.j = j;
            //            hec.e_ij   = p - q;
            hec.e_ij   = p.translation() - q.translation();
            hec.weight = wReg;
            constraints.push_back(hec);
        }
    }
}

void ArapProblem::saveToMesh(ArabMesh& mesh)
{
    for (int i = 0; i < n; ++i)
    {
        auto v            = mesh.vertex_handle(i);
        auto pd           = vertices[i].translation();
        OpenMesh::Vec3f p = {pd[0], pd[1], pd[2]};
        mesh.point(v)     = p;
    }
}
}  // namespace Saiga
