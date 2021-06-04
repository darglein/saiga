/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "ArapProblem.h"

namespace Saiga
{
#ifdef SAIGA_USE_OPENMESH
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


            if (i < j)
            {
                HalfEdgeConstraint hec;
                hec.ids    = {i, j};
                hec.e_ij   = p.translation() - q.translation();
                hec.weight = wReg;
                constraints.push_back(hec);
            }
        }
    }

    std::sort(constraints.begin(), constraints.end());
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
#endif

double ArapProblem::density()
{
    double n = constraints.size() + vertices.size();
    double N = double(vertices.size()) * vertices.size();
    return n / N;
}

void ArapProblem::makeTest()
{
    vertices.emplace_back(Quat::Identity(), Vec3(0, 0, 0));
    vertices.emplace_back(Quat::Identity(), Vec3(1, 0, 0));
    vertices.emplace_back(Quat::Identity(), Vec3(1, 0, 1));

    HalfEdgeConstraint hec;
    hec.weight = wReg;

    hec.ids  = {0, 1};
    hec.e_ij = vertices[0].translation() - vertices[1].translation();
    constraints.push_back(hec);

    hec.ids  = {0, 2};
    hec.e_ij = vertices[0].translation() - vertices[2].translation();
    constraints.push_back(hec);

    //    hec.ids  = {1, 2};
    //    hec.e_ij = vertices[1].translation() - vertices[2].translation();
    //    constraints.push_back(hec);


    {
        int id = 0;
        // add an offset to the first vertex
        auto p = vertices[id].translation();
        target_indices.push_back(id);
        target_positions.push_back(p + Vec3(0, 0.2, 0));
    }

    {
        int id = 1;
        // add an offset to the first vertex
        auto p = vertices[id].translation();
        target_indices.push_back(id);
        target_positions.push_back(p + Vec3(0, -0.2, 0));
    }
}

void ArapProblem::makeSmall(int count)
{
    if (count > (int)vertices.size())
    {
        vertices.resize(count);
    }

    // remove all edges with an invalid vertex
    constraints.erase(std::remove_if(constraints.begin(), constraints.end(),
                                     [&](const HalfEdgeConstraint& c) {
                                         if (c.ids.second >= count) return true;
                                         return false;
                                     }),
                      constraints.end());
}

double ArapProblem::chi2()
{
    double chi2 = 0;

    for (int k = 0; k < (int)target_indices.size(); ++k)
    {
        int i    = target_indices[k];
        auto p   = vertices[i];
        auto t   = target_positions[k];
        Vec3 res = p.translation() - t;
        auto c   = res.squaredNorm();
        chi2 += c;
    }

    for (size_t k = 0; k < constraints.size(); ++k)
    {
        auto& e = constraints[k];
        int i   = e.ids.first;
        int j   = e.ids.second;


        auto pHat = vertices[i];
        auto qHat = vertices[j];
        {
            Vec3 R_eij = pHat.so3() * e.e_ij;
            Vec3 res   = sqrt(e.weight) * (pHat.translation() - qHat.translation() - R_eij);
            auto c     = res.squaredNorm();
            chi2 += c;
        }
        {
            Vec3 R_eji = qHat.so3() * (-e.e_ij);
            Vec3 res   = sqrt(e.weight) * (qHat.translation() - pHat.translation() - R_eji);
            auto c     = res.squaredNorm();
            chi2 += c;
        }
    }
    return 0.5 * chi2;
}
}  // namespace Saiga
