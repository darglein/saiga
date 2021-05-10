/**
 * Copyright (c) 2021 Darius Rückert
 * Licensed under the MIT License.
 * See LICENSE file for more information.
 */

#include "PoseGraph.h"

#include "saiga/core/imgui/imgui.h"
#include "saiga/vision/util/Random.h"

#include <fstream>
#include <algorithm>
namespace Saiga
{
PoseGraph::PoseGraph(const Scene& scene, int minEdges)
{
    vertices.reserve(scene.images.size());
    for (auto& p : scene.images)
    {
        PoseVertex pv;
        pv.SetPose(p.se3.inverse());
        pv.constant = p.constant;
        pv.constant = false;
        vertices.push_back(pv);
    }


    int n = scene.images.size();
    std::vector<std::vector<int>> schurStructure;
    schurStructure.clear();
    schurStructure.resize(n, std::vector<int>(n, 0));
    for (auto& wp : scene.worldPoints)
    {
        for (auto& ref : wp.stereoreferences)
        {
            for (auto& ref2 : wp.stereoreferences)
            {
                int i1 = ref.first;
                int i2 = ref2.first;
                schurStructure[i1][i2]++;
                schurStructure[i2][i1]++;
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
                e.setRel(vertices[i].Pose(), vertices[j].Pose());
                edges.push_back(e);
            }
        }
    }
    sortEdges();
}

void PoseGraph::addNoise(double stddev)
{
    for (auto& e : vertices)
    {
        if (e.constant) continue;
        //        e.T_w_i.se3().translation() += Random::MatrixGauss<Vec3>(0, stddev);
        e.T_w_i.se3() = Random::JitterPose(e.T_w_i.se3(), stddev, stddev);


        //        Quat q = e.se3.unit_quaternion();
        //        Quat q = e.se3.rxso3().quaternion();
        //        q.coeffs() += Random::MatrixGauss<Vec4>(0, stddev);
        //        q.normalize();
        //        e.se3.setQuaternion(q);
    }
}

PoseEdge::TangentType PoseGraph::residual6(const PoseEdge& edge)
{
    return edge.residual(vertices[edge.from].Sim3Pose(), vertices[edge.to].Sim3Pose());
}

double PoseGraph::density()
{
    return double((edges.size() * 2) + vertices.size()) / double(vertices.size() * vertices.size());
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
    std::ofstream strm(file);
    SAIGA_ASSERT(strm.is_open());
    strm.precision(20);
    strm << std::scientific;

    strm << vertices.size() << " " << edges.size() << " " << (int)fixScale << std::endl;
    for (auto& e : vertices)
    {
        strm << e.constant << " " << e.T_w_i.params().transpose() << std::endl;
    }
    for (auto& e : edges)
    {
        strm << e.from << " " << e.to << " " << e.weight << std::endl;
        strm << e.T_i_j.params().transpose() << std::endl;
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
    int num_vertices, num_edges, _fixScale;
    strm >> num_vertices >> num_edges >> _fixScale;
    fixScale = _fixScale;
    vertices.resize(num_vertices);
    edges.resize(num_edges);


    for (auto& e : vertices)
    {
        Eigen::Map<Sophus::Vector<double, 8>> v2(e.T_w_i.data());
        Sophus::Vector<double, 8> v;
        strm >> e.constant >> v;
        v2 = v;
    }

    for (auto& e : edges)
    {
        Sophus::Vector<double, 8> from_v;
        strm >> e.from >> e.to >> e.weight;
        strm >> from_v;
        e.T_i_j = DSim3(from_v);
    }
}

void PoseGraph::AddVertexEdge(int from, int to, double weight)
{
    SAIGA_ASSERT(from != to);
    PoseEdge pe;
    pe.from   = from;
    pe.to     = to;
    pe.weight = weight;
    pe.setRel(vertices[from].Pose(), vertices[to].Pose());
    edges.push_back(pe);
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

void PoseGraph::transform(const SE3& T)
{
    //    for (auto& v : vertices)
    //    {
    //        v.T_w_i.se3 = T * v.T_w_i.se3;
    //    }
    //    for (auto& e : edges)
    //    {
    //        e.T_i_j.se3 = T * e.T_i_j.se3;
    //    }
}

void PoseGraph::invertPoses()
{
    for (auto& v : vertices)
    {
        v.T_w_i = v.T_w_i.inverse();
    }
    for (auto& e : edges)
    {
        e.T_i_j = e.T_i_j.inverse();
    }
}

bool PoseGraph::imgui()
{
    ImGui::PushID(2836759);
    bool changed = false;

    ImGui::Checkbox("fixScale", &fixScale);
    if (ImGui::Button("RMS"))
    {
        rms();
    }
    if (ImGui::Button("invertPoses"))
    {
        invertPoses();
        changed = true;
    }
    if (ImGui::Button("Random Transform"))
    {
        transform(Random::randomSE3());
        changed = true;
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
    strm << " Poses: " << pg.vertices.size() << std::endl;
    strm << " Edges: " << pg.edges.size() << std::endl;
    strm << " Rms: " << pg.rms() << std::endl;
    strm << " Chi2: " << pg.chi2() << std::endl;
    strm << " Density: " << pg.density() * 100 << "%" << std::endl;

    int constantNodes = 0;
    for (auto e : pg.vertices)
        if (e.constant) constantNodes++;
    strm << " Constant Poses: " << constantNodes << std::endl;

    return strm;
}

}  // namespace Saiga
