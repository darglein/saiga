/**
 * Copyright (c) 2017 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "PoseGraph.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/vision/Random.h"

#include <fstream>
namespace Saiga
{
PoseGraph::PoseGraph(const Scene& scene, int minEdges)
{
    poses.reserve(scene.extrinsics.size());
    for (auto& p : scene.extrinsics)
    {
        PoseVertex pv;
        pv.se3      = p.se3;
        pv.constant = p.constant;
        poses.push_back(pv);
    }


    int n = scene.extrinsics.size();
    std::vector<std::vector<int>> schurStructure;
    schurStructure.clear();
    schurStructure.resize(n, std::vector<int>(n, -1));
    for (auto& wp : scene.worldPoints)
    {
        for (auto& ref : wp.stereoreferences)
        {
            for (auto& ref2 : wp.stereoreferences)
            {
                int i1                 = ref.first;
                int i2                 = ref2.first;
                schurStructure[i1][i2] = i2;
                schurStructure[i2][i1] = i1;
            }
        }
    }


    for (int i = 0; i < n; ++i)
    {
        for (int j = 0; j < i; ++j)
        {
            if (schurStructure[i][j] >= minEdges)
            {
                PoseEdge e;
                e.from = i;
                e.to   = j;
                e.setRel(poses[i].se3, poses[j].se3);
                edges.push_back(e);
            }
        }
    }
    sortEdges();
}

void PoseGraph::addNoise(double stddev)
{
    for (auto& e : poses)
    {
        e.se3.translation() += Random::gaussRandMatrix<Vec3>(0, stddev);
        Quat q = e.se3.unit_quaternion();
        q.coeffs() += Random::gaussRandMatrix<Vec4>(0, stddev);
        q.normalize();
        //        e.se3.setQuaternion(q);
    }
}

Vec6 PoseGraph::residual6(const PoseEdge& edge)
{
    auto& _from = poses[edge.from].se3;
    auto& _to   = poses[edge.to].se3;
#ifdef LSD_REL
    auto error_ = _from.inverse() * _to * edge.meassurement.inverse();
#else
    auto error_ = edge.meassurement.inverse() * _to * _from.inverse();
#endif
    return error_.log() * edge.weight;
}

double PoseGraph::density()
{
    return double((edges.size() * 2) + poses.size()) / double(poses.size() * poses.size());
}

double PoseGraph::chi2()
{
    double error = 0;
    for (PoseEdge& e : edges)
    {
        double sqerror;
        sqerror = residual6(e).squaredNorm();
        error += sqerror;
    }
    return error;
}

void PoseGraph::save(const std::string& file)
{
    std::cout << "Saving PoseGraph to " << file << "." << std::endl;
    std::cout << "chi2 " << chi2() << std::endl;
    std::ofstream strm(file);
    SAIGA_ASSERT(strm.is_open());
    strm.precision(20);
    strm << std::scientific;

    strm << poses.size() << " " << edges.size() << std::endl;
    for (auto& e : poses)
    {
        strm << e.constant << " " << e.se3.params().transpose() << std::endl;
    }
    for (auto& e : edges)
    {
        strm << e.from << " " << e.to << " " << e.weight << " " << e.meassurement.params().transpose() << std::endl;
    }
}

void PoseGraph::load(const std::string& file)
{
    std::cout << "Loading scene from " << file << "." << std::endl;


    std::ifstream strm(file);
    SAIGA_ASSERT(strm.is_open());


    auto consumeComment = [&]() {
        while (true)
        {
            auto c = strm.peek();
            if (c == '#')
            {
                std::string s;
                std::getline(strm, s);
            }
            else
            {
                break;
            }
        }
    };


    consumeComment();
    int num_vertices, num_edges;
    strm >> num_vertices >> num_edges;
    poses.resize(num_vertices);
    edges.resize(num_edges);
    std::cout << "Vertices/Edges: " << num_vertices << "/" << num_edges << std::endl;

    for (auto& e : poses)
    {
        Eigen::Map<Sophus::Vector<double, SE3::num_parameters>> v2(e.se3.data());
        Sophus::Vector<double, SE3::num_parameters> v;
        strm >> e.constant >> v;
        v2 = v;
    }

    for (auto& e : edges)
    {
        Eigen::Map<Sophus::Vector<double, SE3::num_parameters>> v2(e.meassurement.data());
        Sophus::Vector<double, SE3::num_parameters> v;
        strm >> e.from >> e.to >> e.weight >> v;
        v2 = v;

        //        e.setRel(poses[e.from].se3, poses[e.to].se3);
    }
    std::sort(edges.begin(), edges.end());
    sortEdges();
}

void PoseGraph::sortEdges()
{
    // first swap if j > i
    for (auto& e : edges)
    {
        if (e.from > e.to) e.invert();
    }

    // and then sort by from/to index
    std::sort(edges.begin(), edges.end());
}

bool PoseGraph::imgui()
{
    ImGui::PushID(2836759);
    bool changed = false;

    if (ImGui::Button("RMS"))
    {
        rms();
    }


    static float sigma = 0.01;
    ImGui::InputFloat("sigma", &sigma);



    if (ImGui::Button("Add Noise"))
    {
        addNoise(sigma);
        changed = true;
    }

    ImGui::PopID();
    return changed;
}

std::ostream& operator<<(std::ostream& strm, PoseGraph& pg)
{
    strm << "[PoseGraph]" << std::endl;
    strm << " Poses: " << pg.poses.size() << std::endl;
    strm << " Edges: " << pg.edges.size() << std::endl;
    strm << " Rms: " << pg.rms() << std::endl;
    strm << " Chi2: " << pg.chi2() << std::endl;
    strm << " Density: " << pg.density() * 100 << "%" << std::endl;
    return strm;
}

}  // namespace Saiga
